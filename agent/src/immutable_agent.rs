use crate::exec_python::*;
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
use crate::nous_structs::*;
use crate::utils::*;
use crate::webscraper_hook::*;
use crate::{
    CODE_PYTHON_SYSTEM_MESSAGE,
    FURTER_TASK_BY_TOOLCALL_PROMPT,
    IS_TERMINATION_SYSTEM_PROMPT,
    ITERATE_CODING_FAIL_TEMPLATE,
    ITERATE_CODING_INCORRECT_TEMPLATE,
    ITERATE_CODING_START_TEMPLATE,
};
use anyhow;
use chrono::Utc;
// use serde::{ Deserialize, Serialize };
use serde_json::Value;

use once_cell::sync::Lazy;

pub static GROUNDING_CHECK_TEMPLATE: Lazy<String> = Lazy::new(|| {
    let today = Utc::now().format("%Y-%m-%dT").to_string();
    format!(r#"
<|im_start|>system You are an AI assistant. Your task is to determine whether a question requires grounding in real-world date, time, location, or physics.

When given a task, please follow these steps to think it through and then act:

Identify Temporal Relevance: Determine if the question requires current or time-sensitive information. Note that today's date is {}.
Check for Location Specificity: Identify if the question is location-specific.
Determine Real-time Data Dependency: Assess if the answer depends on real-time data or specific locations.
Suggest Grounding Information: If grounding is needed, suggest using today's date to cross-validate the reply. Otherwise, suggest reliable sources to obtain the necessary grounding data.

Remember that you are a dispatcher; you DO NOT work on tasks yourself. Your role is to direct the process.

In your reply, list out your think-aloud steps clearly:

Example 1:

When tasked with "What is the weather like in New York?" reshape your answer as follows:

{{
    \"my_thought_process\": [
        \"my_goal: goal is to identify whether grounding is needed for this task\",
        \"Determine if the question requires current or time-sensitive information: YES\",
        \"Provide assistance to agent for this task grounding information: today's date is xxxx-xx-xx\",
        \"Echo original input verbatim in key_points section\"
    ],
    \"key_points\": [\"today's date is xxxx-xx-xx, What is the weather like in New York\"]
}}

Example 2:

When tasked with \"Who killed John Lennon?\" reshape your answer as follows:

{{
    \"my_thought_process\": [
        \"my_goal: goal is to identify whether grounding is needed for this task\",
        \"Determine if the question requires current or time-sensitive information: NO\",
        \"Echo original input verbatim in key_points section\"
    ],
    \"key_points\": [\"Who killed John Lennon?\"]
}}

Use this format for your response:

```json
{{
    \"my_thought_process\": [
        \"my_goal: goal is to identify whether grounding is needed for this task\",
        \"thought_process_one: my judgement at this step\",
        \"...\",
        \"thought_process_N: my judgement at this step\"
    ],
    \"grounded_or_not\": \"YES\" or \"NO\",
    \"key_points\": [\"point1\", \"point2\", ...]
}}
"#, today)
});

const INTERNAL_ROUTING_PROMPT: &'static str =
    r#"
You are a helpful AI assistant acting as a task dispatcher. Below are several paths that an agent can take and their abilities. Examine the task instruction and the current result, then decide whether the task is complete or needs further work. If further work is needed, dispatch the task to one of the agents. Please also extract key points from the current result. The descriptions of the agents are as follows:

1. **coding_python**: Specializes in generating clean, executable Python code for various tasks.
2. **user_proxy**: Represents the user by delegating tasks to agents, reviewing their outputs, and ensuring tasks meet user requirements; it is also responsible for receiving final task results.

Use this format to reply:
```json
{{
    "continue_or_terminate": "TERMINATE" or "CONTINUE",
    "next_task_handler": "some_task_handler" (leave empty if "TERMINATE"),
    "key_points": ["point1", "point2", ...]
}
```
Dispatch to user_proxy when all tasks are complete.
"#;

const NEXT_STEP_PLANNING: &'static str =
    r#"
    You are a helpful AI assistant with extensive capabilities. Your goal is to help complete tasks and create plausible answers grounded in real-world history of events and physics with minimal steps.

    You have three built-in tools to solve problems:
    
    use_intrinsic_knowledge: You can answer many questions and provide a wealth of knowledge from within yourself. This should be your first approach to problem-solving.
    code_with_python: Generates and executes Python code for various tasks based on user input. It can handle mathematical computations, data analysis, large datasets, complex operations through optimized algorithms, providing precise, deterministic outputs.
    search_with_bing: Performs an internet search using Bing and returns relevant results based on a query. Use it to get information you don't have or cross-check for real-world grounding.
    
    When given a task, follow these steps:
    
    Determine whether the task can be completed in a single step with your built-in tools.
    If yes, consider this as special cases for one-step completion which should be placed in the "steps_to_take" section.
    If determined that it can be answered with intrinsic knowledge:
    DO NOT try to answer it yourself.
    Pass the task to the next agent by using the original input text verbatim as one single step in the "steps_to_take" section.
    If neither intrinsic knowledge nor built-in tools suffice:
    Strategize and outline necessary steps to achieve the final goal.
    Each step corresponds to a task that can be completed with one of three approaches: intrinsic knowledge, creating Python code, or searching with Bing.
    You don't need to do grounding check for well documented, established facts when there is no direct or inferred reference point of date or locality in task.
    When listing steps:
    Think about why you outlined such a step.
    When cascading down to coding tasks:
    Constrain them ideally into one coding task.
    Fill out the "steps_to_take" section of your reply template accordingly.
    
    In your reply, list out your think-aloud steps clearly:
    
    Example 1:
    
    When tasked with "calculate prime numbers up to 100," reshape your answer as follows:
    
    {
        "my_goal": "goal is to help complete this mathematical computation efficiently",
        "my_thought_process": [
            "Determine if this task can be done in single step: NO",
            "Determine if this task can be done with coding: YES",
            "Strategize on breaking down into logical subtasks: ",
            "[Define function checking if number is prime]",
            "[Loop through numbers 2-100 calling function]",
            "[Print each number if it's prime]",
            "...",
            "[Check for unnecessary breakdowns especially for 'coding' tasks]: merge into single coding action"
        ],
        "steps_to_take": ["Define function checking primes; loop through 2-100 calling function; print primes"]
    }
    
    Example 2:
    
    When tasked with "find out how old Barack Obama is" reshape your answer as follows:
    
    {
        "my_goal": "goal is finding Barack Obama's current age quickly",
        "my_thought_process": [
            "Determine if this task can be done in single step: YES",
            "Can be answered via intrinsic knowledge directly: YES",
            "check real world grounding: my knowledge base is based on data grounded in 2022; need current year",
            "use search_with_bing tool finding current year",
            "collate age based on birth year (1961) and current year"
        ],
        "steps_to_take": ["Use 'search_with_bing' tool finding current year", 
                          "Calculate Barack Obama's age from birth year (1961)"]
    }

    Example 3:
    
    When tasked with "find out when Steve Jobs died," reshape your answer as follows:
    
    {
        "my_goal": "goal is finding Steve Jobs' date of death accurately",
        "my_thought_process": [
           "Determine if this task could utilize built-in tools: YES, can use intrinsic knowledge"
         ],
         "steps_to_take": ["find out when Steve Jobs died"]
    }

    Example 4 (Fact):

When tasked with "how to describe Confucius" reshape your answer as follows:

{
    "my_goal": "goal is to provide an accurate description of Confucius",
    "my_thought_process": [
       "Determine if this task can be done in a single step: YES",
       "Can it utilize intrinsic knowledge directly? YES"
       "Confucius was a historical figure whose details are well-documented: no need to check grounding"
       ]
      ],
      "steps_to_take": ["how to describe Confucius"]
}

Use this format for your response:
```json
{
    "my_thought_process": [
        "thought_process_one: my judgement at this step",
        "...",
        "though_process_N: : my judgement at this step"
    ],
    "steps_to_take": ["Step description", "..."] 
}
```
"#;

const NEXT_STEP_BY_TOOLCALL: &'static str =
    r#"
<|im_start|>system You are a function-calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Do not make assumptions about what values to plug into functions.

Here are the available tools:

<tools>
1. **use_intrinsic_knowledge**: 
Description: Solves tasks using capabilities and knowledge obtained at trainning time, the carveate is that it is frozen by the cut-off date and it's not aware of real world date of its operation.
Example Call:
<tool_call>
{"arguments": {"task": "tell a joke"}, 
"name": "use_intrinsic_knowledge"}
</tool_call>

2. **search_with_bing**: 
Description: Conducts an internet search using Bing search engine and returns relevant results based on the query provided by the user. It's a safe choice to try searching for results; if they are not satisfactory, you can use suspect URLs from these search results with "get_webpage_text" function.

Special Note 1: This function performs an internet search to find relevant webpages based on your query. It helps narrow down potential sources of information before extracting specific content.

Special Note 2: Using search_with_bing as an initial step can make subsequent tasks more targeted by providing exact links that can then be scraped using get_webpage_text. This approach ensures higher relevance and accuracy of retrieved data.
Example Call:
<tool_call>
{"arguments": {"query": "latest AI research trends"}, 
"name": "search_with_bing"}
</tool_call>

3. **code_with_python**: 
Description: Generates clean, executable Python code for various tasks based on user input.
Example Call:
<tool_call>
{"arguments": {"key_points": "Create a Python script that reads a CSV file and plots a graph"}, 
"name": "code_with_python"}
</tool_call>

4. **get_webpage_text**: 
Description: Fetches all textual content from the specified webpage URL and returns it as plain text. It does not parse or structure the data in any specific format, so it may include extraneous information such as navigation menus, advertisements, and other non-essential text elements present on the page.

Special Note 1: This function retrieves raw text from a given URL without filtering out irrelevant content. Therefore, using a URL that is not unique to your solution may result in obtaining unrelated data.

Special Note 2: While this function can extract text from a known relevant webpage directly, it is often more effective to first use search_with_bing to find precise URLs before scraping them for targeted information.

Example Call:
<tool_call>
{"arguments": {"url": "https://example.com"}, 
"name": "get_webpage_text"}
</tool_call>
</tools>

Function Definitions:

use_intrinsic_knowledge
Description: Solves tasks using built-in capabilities.
Parameters:
problem: The task you receive (type:string)
Required Parameters:["task"]

search_with_bing
Description: Conducts an internet search using Bing search engine and returns relevant results based on the query provided by the user.
Parameters:
query: The search query to be executed on Bing (type:string)
Required Parameters:["query"]

code_with_python
Description Generates clean executable Python code for various tasks.Parameters key_points Key points describing what kind of problem needs to be solved with Python code(type:string) Required Parameters:["key_points"]

get_webpage_text
Description Retrieves all textual content froma specified website URL.It does not parse or structure data in any specific format; hence,it may include extraneous information such as navigation menus advertisements,and other non-essential text elements present onthe page.Parameters url The URLofthe website from which to fetch textual content(type:string) Required Parameters:["url"]

Remember that you area dispatcher;you DO NOT workon tasks yourself.

Examples of tool calls for different scenarios:

To handlea task like "tell a joke" with intrinsic knowledge: <tool_call>{"
arguments": {"task": "tell ajoke"},
"name": "use_intrinsic_knowledge"}</tool_call>
To retrieve webpage text: <tool_call>{"
arguments": {"url": "https://example.com"},
"name": "get_webpage_text"}</tool_call>
To generate Python code: <tool_call>{"arguments": {"key_points": "CreateaPython script that reads data from an API and stores it in adatabase"},
"name": "code_with_python"}</tool_call>
To perform an internet search: <tool_call>{"
arguments": {"query": "best practices in software development"},
"name": "search_with_bing"}</tool_call>

For each function call, return a JSON object with function name and arguments within <tool_call></tools> XML tags as follows:

<tools>  
{"arguments": <args-dict>,   
"name": "<function_name>"}
</tools>
"#;

// pub fn som() {
//     let system_message = ChatCompletionRequestMessage::new_system_message(system_prompt, None);

//     chat_request.messages.push(system_message);
//     let user_message = ChatCompletionRequestMessage::new_user_message(
//         ChatCompletionUserMessageContent::Text(user_input.to_string()),
//         None
//     );

//     chat_request.messages.push(user_message);
// }

pub struct ImmutableAgent {
    pub name: String,
    pub system_prompt: String,
    pub llm_config: Option<Value>,
    pub tools_map_meta: String,
    pub description: String,
}

impl ImmutableAgent {
    pub fn simple(name: &str, system_prompt: &str) -> Self {
        ImmutableAgent {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
            llm_config: None,
            tools_map_meta: String::from(""),
            description: String::from(""),
        }
    }

    pub fn new(
        name: &str,
        system_prompt: &str,
        llm_config: Option<Value>,
        tools_map_meta: &str,
        description: &str
    ) -> Self {
        ImmutableAgent {
            name: name.to_string(),
            system_prompt: system_prompt.to_string(),
            llm_config,
            tools_map_meta: tools_map_meta.to_string(),
            description: description.to_string(),
        }
    }

    pub async fn get_user_feedback(&self) -> String {
        use std::io::{ self, Write };
        print!("User input: ");

        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();

        io::stdin().read_line(&mut input).expect("Failed to read line");

        if let Some('\n') = input.chars().next_back() {
            input.pop();
        }
        if let Some('\r') = input.chars().next_back() {
            input.pop();
        }

        return input;
    }

    pub async fn furter_task_by_toolcall(
        &self,
        chat_request: &mut ChatCompletionRequest,
        input: &str
    ) -> Option<String> {
        let output: NousResponseMessage = chat_completions_full(
            chat_request,
            &FURTER_TASK_BY_TOOLCALL_PROMPT,
            input
        ).await.expect("Failed to generate reply");

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
        input: &str
    ) -> Option<String> {
        let output: NousResponseMessage = chat_completions_full(
            chat_request,
            &NEXT_STEP_BY_TOOLCALL,
            &input
        ).await.expect("Failed to generate reply");

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

                        let steps_vec = self.planning(chat_request, &task).await;

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
    pub async fn planning(
        &self,
        chat_request: &mut ChatCompletionRequest,
        input: &str
    ) -> Vec<String> {
        let output: NousResponseMessage = chat_completions_full(
            chat_request,
            &NEXT_STEP_PLANNING,
            input
        ).await.expect("Failed to generate reply");

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
        task_vec: &Vec<String>
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
            res = self.furter_task_by_toolcall(chat_request, &initial_input).await.unwrap();
            initial_input = match task_vec.pop() {
                Some(s) =>
                    format!(
                        "Here is the result from previous step: {}, here is the next task: {}",
                        res,
                        s
                    ),
                None => {
                    break;
                }
            };
        }
        Ok(res)
    }

    pub async fn choose_next_step_and_(
        &self,
        chat_request: &mut ChatCompletionRequest,
        current_text_result: &str,
        instruction: &str
    ) -> (bool, Option<String>, String) {
        let user_prompt = format!(
            "Given the task: {:?}, examine current result: {}, please decide whether the task is done or need further work",
            instruction,
            current_text_result
        );

        let raw_reply = chat_completions_full(
            chat_request,
            &INTERNAL_ROUTING_PROMPT,
            &user_prompt
        ).await.expect("llm generation failure");
        println!("{:?}", raw_reply.content_to_string().clone());
        let (stop_here, speaker, key_points) = parse_next_move_and_(
            &raw_reply.content_to_string(),
            Some("next_task_handler")
        );

        // let _ = save_message(conn, &self.name, &key_points, &speaker).await;
        (stop_here, speaker, key_points.join(","))
    }

    pub async fn _is_termination(
        &self,
        chat_request: &mut ChatCompletionRequest,
        current_text_result: &str,
        instruction: &str
    ) -> (bool, String) {
        let user_prompt = format!(
            "Given the task: {:?}, examine current result: {}, please decide whether the task is done or not",
            instruction,
            current_text_result
        );

        println!("{:?}", user_prompt.clone());

        let raw_reply = chat_completions_full(
            chat_request,
            &IS_TERMINATION_SYSTEM_PROMPT,
            &user_prompt
        ).await.expect("llm generation failure");

        println!("_is_termination raw_reply: {:?}", raw_reply.content_to_string());

        let (terminate_or_not, _, key_points) = parse_next_move_and_(
            &raw_reply.content_to_string(),
            None
        );

        (terminate_or_not, key_points.join(","))
    }

    pub async fn code_with_python(
        &self,
        chat_request: &mut ChatCompletionRequest,
        message_text: &str
    ) -> anyhow::Result<()> {
        let formatter = ITERATE_CODING_START_TEMPLATE.lock().unwrap();
        let user_prompt = formatter(&[message_text]);

        for n in 1..9 {
            println!("Iteration: {}", n);
            match
                chat_completions_full(
                    chat_request,
                    &CODE_PYTHON_SYSTEM_MESSAGE,
                    &user_prompt
                ).await?.content
            {
                NousContent::Text(_out) => {
                    // let head: String = _out.chars().take(200).collect::<String>();
                    println!("Raw generation {n}:\n {}\n\n", _out.clone());
                    let (this_round_good, code, exec_result) = run_python_wrapper(&_out).await;
                    println!("code:\n{}\n\n", code.clone());
                    println!("Run result {n}: {}\n", exec_result.clone());

                    if this_round_good {
                        let (terminate_or_not, key_points) = self._is_termination(
                            chat_request,
                            &exec_result,
                            &user_prompt
                        ).await;
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
