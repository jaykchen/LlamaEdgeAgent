use crate::exec_python::*;
use crate::nous_structs::*;
use crate::utils::*;
use crate::webscraper_hook::*;
use crate::{
    CODE_PYTHON_PROMPT, FURTER_TASK_BY_TOOLCALL_PROMPT, GROUNDING_CHECK_TEMPLATE,
    IS_TERMINATION_PROMPT, ITERATE_CODING_FAIL_TEMPLATE, ITERATE_CODING_INCORRECT_TEMPLATE,
    ITERATE_CODING_START_TEMPLATE, NEXT_STEP_BY_TOOLCALL_PROMPT, NEXT_STEP_PLANNING_PROMPT,
};
use anyhow;
use endpoints::{
    chat::{
        ChatCompletionRequest,
        // ChatCompletionObject,
        // ChatCompletionRequestMessage,
        // ChatCompletionRole,
        // ChatCompletionSystemMessage,
        // ChatCompletionUserMessage,
        // ChatCompletionUserMessageContent,
    },
    // common::Usage,
};

pub struct ImmutableAgent {
    pub name: String,
    pub system_prompt: String,
}

impl ImmutableAgent {
    pub fn new(name: &str, system_prompt: &str) -> Self {
        ImmutableAgent {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
        }
    }

    pub async fn get_user_feedback(&self) -> String {
        use std::io::{self, Write};
        print!("User input: ");

        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();

        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        if let Some('\n') = input.chars().next_back() {
            input.pop();
        }
        if let Some('\r') = input.chars().next_back() {
            input.pop();
        }

        if input == "stop" {
            std::process::exit(0);
        }
        return input;
    }

    pub async fn furter_task_by_toolcall(
        &self,
        chat_request: &mut ChatCompletionRequest,
        input: &str,
    ) -> Option<String> {
        let output: NousResponseMessage =
            chat_completions_full(chat_request, &FURTER_TASK_BY_TOOLCALL_PROMPT, input)
                .await
                .expect("Failed to generate reply");

        match &output.content {
            NousContent::Text(t) => {
                return Some(t.to_string());
            }
            NousContent::NousToolCall(call) => {
                let args = call.clone().arguments.unwrap_or_default();

                let res = match call.name.as_str() {
                    "get_webpage_text" => {
                        let url = args
                            .get("url")
                            .ok_or_else(|| anyhow::anyhow!("Missing 'url' argument"))
                            .ok()?
                            .to_string();

                        get_webpage_text(url).await.ok()?
                    }
                    "search_with_bing" => {
                        let query = args
                            .get("query")
                            .ok_or_else(|| anyhow::anyhow!("Missing 'query' argument"))
                            .ok()?
                            .to_string();
                        search_with_bing(&query).await.ok()?
                    }
                    "code_with_python" => {
                        let key_points = args
                            .get("key_points")
                            .ok_or_else(|| anyhow::anyhow!("Missing 'key_points' argument"))
                            .ok()?
                            .to_string();
                        // let _ = self.code_with_python(&key_points).await;

                        String::from("code is being generated")
                    }
                    _ => {
                        return None;
                    }
                };
                Some(res)
            }
        }
    }

    pub async fn next_step_by_toolcall(
        &self,
        chat_request: &mut ChatCompletionRequest,
        input: &str,
    ) -> Option<String> {
        let output: NousResponseMessage =
            chat_completions_full(chat_request, &NEXT_STEP_BY_TOOLCALL_PROMPT, &input)
                .await
                .expect("Failed to generate reply");

        match &output.content {
            NousContent::Text(_) => {
                todo!();
            }
            NousContent::NousToolCall(call) => {
                let args = call.clone().arguments.unwrap_or_default();

                let res = match call.name.as_str() {
                    "use_intrinsic_knowledge" => {
                        let task = args
                            .get("task")
                            .ok_or_else(|| anyhow::anyhow!("Missing 'task' argument"))
                            .ok()?
                            .to_string();

                        let steps_vec = self.next_step_planning(chat_request, &task).await;

                        let _ = self.stepper(chat_request, &steps_vec).await;
                        std::process::exit(0);
                    }
                    "get_webpage_text" => {
                        let url = args
                            .get("url")
                            .ok_or_else(|| anyhow::anyhow!("Missing 'url' argument"))
                            .ok()?
                            .to_string();

                        get_webpage_text(url).await.ok()?
                    }
                    "search_with_bing" => {
                        let query = args
                            .get("query")
                            .ok_or_else(|| anyhow::anyhow!("Missing 'query' argument"))
                            .ok()?
                            .to_string();
                        search_with_bing(&query).await.ok()?
                    }
                    "code_with_python" => {
                        let key_points = args
                            .get("key_points")
                            .ok_or_else(|| anyhow::anyhow!("Missing 'key_points' argument"))
                            .ok()?
                            .to_string();
                        let _ = self.code_with_python(chat_request, &key_points).await;

                        String::from("code is being generated")
                    }

                    _ => {
                        return None;
                    }
                };
                Some(res)
            }
        }
    }
    pub async fn next_step_planning(
        &self,
        chat_request: &mut ChatCompletionRequest,
        input: &str,
    ) -> Vec<String> {
        let output: NousResponseMessage =
            chat_completions_full(chat_request, &NEXT_STEP_PLANNING_PROMPT, input)
                .await
                .expect("Failed to generate reply");

        match &output.content {
            NousContent::Text(_out) => {
                println!("{:?}\n\n", _out.clone());
                let mut res = parse_planning_steps(_out);
                res.reverse();
                res
            }
            _ => unreachable!(),
        }
    }

    pub async fn stepper(
        &self,
        chat_request: &mut ChatCompletionRequest,
        task_vec: &Vec<String>,
    ) -> anyhow::Result<String> {
        let mut task_vec = task_vec.clone();
        let mut initial_input = match task_vec.pop() {
            Some(s) => s,
            None => {
                return Err(anyhow::Error::msg("no task to handle"));
            }
        };
        let mut res = String::new();
        loop {
            res = self
                .furter_task_by_toolcall(chat_request, &initial_input)
                .await
                .unwrap();
            initial_input = match task_vec.pop() {
                Some(s) => format!(
                    "Here is the result from previous step: {}, here is the next task: {}",
                    res, s
                ),
                None => {
                    break;
                }
            };
        }
        Ok(res)
    }

    pub async fn _is_termination(
        &self,
        chat_request: &mut ChatCompletionRequest,
        current_text_result: &str,
        instruction: &str,
    ) -> (bool, String) {
        let user_prompt = format!(
            "Given the task: {:?}, examine current result: {}, please decide whether the task is done or not",
            instruction,
            current_text_result
        );

        println!("{:?}", user_prompt.clone());

        let raw_reply = chat_completions_full(chat_request, &IS_TERMINATION_PROMPT, &user_prompt)
            .await
            .expect("llm generation failure");

        println!(
            "_is_termination raw_reply: {:?}",
            raw_reply.content_to_string()
        );

        let (terminate_or_not, _, key_points) =
            parse_next_move_and_(&raw_reply.content_to_string(), None);

        (terminate_or_not, key_points.join(","))
    }

    pub async fn code_with_python(
        &self,
        chat_request: &mut ChatCompletionRequest,
        message_text: &str,
    ) -> anyhow::Result<()> {
        let formatter = ITERATE_CODING_START_TEMPLATE.lock().unwrap();
        let user_prompt = formatter(&[message_text]);

        for n in 1..9 {
            println!("Iteration: {}", n);
            match chat_completions_full(chat_request, &CODE_PYTHON_PROMPT, &user_prompt)
                .await?
                .content
            {
                NousContent::Text(_out) => {
                    // let head: String = _out.chars().take(200).collect::<String>();
                    println!("Raw generation {n}:\n {}\n\n", _out.clone());
                    let (this_round_good, code, exec_result) = run_python_wrapper(&_out).await;
                    println!("code:\n{}\n\n", code.clone());
                    println!("Run result {n}: {}\n", exec_result.clone());

                    if this_round_good {
                        let (terminate_or_not, key_points) = self
                            ._is_termination(chat_request, &exec_result, &user_prompt)
                            .await;
                        println!("Termination Check: {}\n", terminate_or_not);
                        // if terminate_or_not {
                        //     println!("key_points: {:?}\n", key_points);

                        //     self.get_user_feedback().await;
                        // }
                    }

                    let formatter = if this_round_good {
                        ITERATE_CODING_INCORRECT_TEMPLATE.lock().unwrap()
                    } else {
                        ITERATE_CODING_FAIL_TEMPLATE.lock().unwrap()
                    };

                    let user_prompt = formatter(&[&code, &exec_result]);
                    // let result_message = Message {
                    //     name: None,
                    //     content: NousContent::Text(user_prompt),
                    //     role: ChatCompletionRole::User,
                    // };

                    // messages.push(result_message);
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}

/* pub async fn compress_chat_history(message_history: &Vec<Message>) -> Vec<Message> {
    let message_history = message_history.clone();
    let (system_messages, messages) = message_history.split_at(2);
    let mut system_messages = system_messages.to_vec();

    let chat_history_text = messages
        .into_iter()
        .map(|m| m.content_to_string())
        .collect::<Vec<String>>()
        .join("\n");

    let messages = vec![
        Message {
            role: ChatCompletionRole::System,
            name: None,
            content: NousContent::Text(FURTER_TASK_BY_TOOLCALL_PROMPT.to_string()),
        },
        Message {
            role: ChatCompletionRole::User,
            name: None,
            content: NousContent::Text(chat_history_text),
        }
    ];

    let max_token = 1000u16;
    let output: NousResponseMessage = chat_completions_full(
        messages.clone(),
        max_token
    ).await.expect("Failed to generate reply");

    match output.content {
        NousContent::Text(compressed) => {
            let message = Message {
                role: ChatCompletionRole::User,
                name: None,
                content: NousContent::Text(compressed),
            };

            system_messages.push(message);
        }
        _ => unreachable!(),
    }

    system_messages
} */
