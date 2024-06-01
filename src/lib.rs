pub mod exec_python;
use exec_python::run_python_capture;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::{ Arc, Mutex };
use serde_json::{Value, json};
use serde::{ Serialize, Deserialize };
use flowsnet_platform_sdk::logger;
use webhook_flows::{
    create_endpoint,
    request_handler,
    route::{ get, post, route, RouteError, Router },
    send_response,
};
use dotenv::dotenv;

type FormatterFn = Box<dyn (Fn(&[&str]) -> String) + Send + Sync>;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    create_endpoint().await;
}

#[request_handler(get)]
async fn handler(
    _headers: Vec<(String, String)>,
    _subpath: String,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>
) {
    dotenv().ok();
    logger::init();
    let mut router = Router::new();
    router.insert("/run", vec![post(run_code_by_post_handler)]).unwrap();
    if let Err(e) = route(router).await {
        match e {
            RouteError::NotFound => {
                send_response(404, vec![], b"No route matched".to_vec());
            }
            RouteError::MethodNotAllowed => {
                send_response(405, vec![], b"Method not allowed".to_vec());
            }
        }
    }
}

async fn run_code_by_post_handler(
    _headers: Vec<(String, String)>,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>
) {
    #[derive(Serialize, Deserialize, Clone, Debug, Default)]
    pub struct AgentLoad {
        pub code: Option<String>,
        pub holder: Option<String>,
        pub text: Option<String>,
    }

    let load: AgentLoad = match serde_json::from_slice(&_body) {
        Ok(obj) => obj,
        Err(_e) => {
            log::error!("failed to parse body: {}", _e);
            return;
        }
    };
    if let Some(code) = load.code {
        let res = run_python_capture(&code);
        log::info!("{:?}", res.clone());
        send_response(
            200,
            vec![
                (String::from("content-type"), String::from("application/json")),
                (String::from("Access-Control-Allow-Origin"), String::from("*"))
            ],
            json!(res).to_string().as_bytes().to_vec()
        );
    }
}

lazy_static! {
    pub static ref USER_PROXY_SYSTEM_PROMPT: String =
        r#"
        You are a helpful AI assistant acting as the user's proxy. You act according to these rules:
1. When you receive an instruction from the user, you dispatch the task to an agent in the pool of agents.
2. When you receive agent's answer, you will judge whether the task has been completed, if not done, dispatch it to an agent to further work on it. If done, mark it as "TERMINATE" and save the result for view by the user.
please also extract key points of the result and put them in your reply in the following format:
    ```json
    {
        "continue_to_work_or_end": "TERMINATE" or "CONTINUE",
        "key_points_of_current_result": "key points"
    }
    ```
    "#.to_string();

    pub static ref IS_TERMINATION_SYSTEM_PROMPT: String =
        r#"
    You are a helpful AI assistant acting as a gatekeeper in a project. You will be given a task instruction and the current result, please decide whether the task is done or not, please also extract key points of current result and put them in your reply in the following format:
    ```json
    {
        "continue_to_work_or_end": "TERMINATE" or "CONTINUE",
        "key_points_of_current_result": "key points"
    }
    ```
    "#.to_string();

    pub static ref SUMMARIZE_CHAT_HISTORY_PROMPT: String =
        r#"
You are an AI language model assisting with an iterative coding task. The chat history contains multiple rounds of code execution, including both successful and failed runs. Your goal is to compress this chat history into a concise summary that highlights key observations and results from each iteration, focusing specifically on why the code failed or produced incorrect results. Be analytical and take a bird's-eye view of the task; your intention is to help the agent gain deeper insights rather than focusing on each individual run. If you identify any patterns, include them at the end of your summary.

The chat history content is wrapped in specific templates as follows:

Incorrect Result Template:

Executed the code below:
{code}
producing the following result, but the result is incorrect:
{result}

Execution Failure Template:

Failed to execute the code:
{code}, got the following errors:
{errors}

When summarizing chat history, ensure to include:

- A brief description of each iteration's attempt.
- Key reasons why each attempt failed or produced an incorrect result.
- Avoid repeating identical mistakes.
- Discard low-relevancy record from history.
- Identify any patterns."#.to_string();

    pub static ref ROUTING_SYSTEM_PROMPT: String =
        r#"
You are a helpful AI assistant acting as a discussion moderator or speaker selector. Below are several agents and their abilities. Examine the task instruction and the current result, then decide whether the task is complete or needs further work. If further work is needed, dispatch the task to one of the agents. Please also extract key points from the current result. The descriptions of the agents are as follows:

1. **coding_agent**: Specializes in generating clean, executable Python code for various tasks.
2. **user_proxy**: Represents the user by delegating tasks to agents, reviewing their outputs, and ensuring tasks meet user requirements; it is also responsible for receiving final task results.

Use this format to reply:
```json
{
    "continue_to_work_or_end": "TERMINATE" or "CONTINUE",
    "next_speaker": "some_speaker" (leave empty if "TERMINATE"),
    "key_points_of_current_result": "key points"
}
```
Dispatch to user_proxy when all tasks are complete.
"#.to_string();

    pub static ref ROUTER_AGENT_SYSTEM_PROMPT: String =
        r#"
You are a helpful AI assistant acting as a discussion moderator or speaker selector. You will read descriptions of several agents and their abilities, examine the task instruction and the current result, and decide whether the task is done or needs further work. The descriptions of the agents are as follows:

1. **router_agent**: Efficiently manages and directs tasks to appropriate agents based on evaluation criteria.
2. **coding_agent**: Specializes in generating clean, executable Python code for various tasks.
3. **user_proxy**: Represents the user by delegating tasks to agents, reviewing their outputs, and ensuring tasks meet user requirements.
        
Use the following format to reply:
```json
{
    "continue_to_work_or_end": "TERMINATE" or "CONTINUE",
    "next_speaker": "some_speaker" or empty in case "TERMINATE" in the previous field
}
```
    "#.to_string();
    // Follow these guidelines:"#.to_string();

    pub static ref PLANNING_SYSTEM_PROMPT: String =
        r#"
You are a helpful AI assistant with extensive capabilities:
You can answer many questions and provide a wealth of knowledge from within yourself.
You have several built-in tools:
- The function "start_coding" generates and executes Python code for various tasks based on user input, it's extremely powerful, it can complete all coding related tasks in one step.
- The function "get_webpage_text" retrieves all text content from a given URL.
- The function "search_bing" performs an internet search using Bing and returns relevant results based on the query provided by the user.

When given a task, please follow these steps to think it through and then act:
1. Determine whether the task can be completed in a single step using your intrinsic knowledge or one of your built-in tools
- Consider this as special cases for one-step completion, which should be placed in the "steps_to_take" section.
- If determined that it can be answered with intrinsic knowledge, DO NOT try to answer it yourself. Pass the task to the next agent by using the original input text verbatim as one single step to fill in the "steps_to_take" section.
2. Gauge whether the task is most likely to be best completed with coding. If so, merge multiple logical sub-steps into one comprehensive step due to available dedicated coding tools.
3. If neither intrinsic knowledge nor built-in tools suffice, strategize and outline up to 3 necessary steps to achieve the final goal.
4. Check whether you've slipped your mind and broken down a "coding task" into baby steps; if so, merge these steps back into one.
5. Fill out the "steps_to_take" section of your reply template.
Remember that you are a dispatcher; you DO NOT work on tasks yourself.

In your reply, list out your think-aloud steps clearly:

Example:
When tasked with "calculate prime numbers up to 100," you should reshape your answer as follows: 
{
    "my_thought_process": [
        "Determine if this task can be done in single step: NO",
        "Determine if this task can be done with coding: YES,
        "Strategize on breaking task into logical subtasks: [
        "Define a function to check if a number is prime or not.",
        "Use a loop iterating through numbers from 2 up-to 100.",
        "Call this function within loop." ]",
        "Check for unnecessary breakdowns especially for 'coding' tasks: made a mistake, should be a coding task",
        "Fill out 'steps_to_take' accordingly: merge steps into one [
        "Define a function that checks if numbers are prime. Use this function within loop iterating through numbers from 2 up-to 100. Print each number if it's prime." ]"
    ],
    "steps_to_take": [ "Define a function that checks if numbers are prime. Use this function within loop iterating through numbers from 2 up-to 100. Print each number if it's prime." ]
}

Example:

When tasked with "find out how old is Barack Obama", you should reshape your answer as follows: 
{
    "my_thought_process": [
        "Determine if this task can be done in single step: YES",
        "Can be answered with intrinsic knowledge: YES, use original input to fill "steps_to_take" section"
    ],
    "steps_to_take": [
    "find out how old is Barack Obama"
    ]
}

Example:

When tasked with "find out when Steve Jobs died", you should reshape your answer as follows:
{
    "my_thought_process": [
        "Determine if this task can be done in single step: YES",
        "Can use built-in tool 'search_bing' directly to get the answer"
    ],
    "steps_to_take": [
    "Use 'search_bing' tool to find the date of Steve Jobs death"
    ]
}

Use this format for your response:
```json
{
    "my_thought_process": [
        "thought_process_one: my judgement at this step",
        "...",
        "though_process_N: : my judgement at this step"
    ],
    "steps_to_take": ["Step description", ...] 
}
```
"#.to_string();

    // For coding tasks specifically:
    // - Always merge multiple steps into one single step since you have dedicated coding tools that can complete such tasks in one go. Do not break down coding tasks into separate sub-steps.

    // Guidelines:
    // - Attempt to solve non-coding related queries using intrinsic knowledge or built-in tools first before resorting to breaking down into further actionable steps.
    // - For non-coding related multi-step problems, outline up to 3 necessary steps clearly and concisely.
    // - Treat coding problems as special cases where multiple logical sub-steps should be merged into one comprehensive step due to available dedicated coding tools.

    pub static ref CODE_PYTHON_SYSTEM_MESSAGE: String =
        r#"`You are a helpful AI assistant.
Provide clean, executable Python code blocks to solve tasks, without adding explanatory sentences. Follow these guidelines:
1. Use Python code blocks to perform tasks such as collecting information, executing operations, or outputting results. Ensure the code is ready to execute without requiring user modifications.
2. Address tasks step by step using Python code. If a plan is necessary, it should be implicit within the code structure.
3. Always use 'print' for outputting results within the Python code.
4. When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest.
5. Do not include multiple code blocks in one response. Ensure each response contains only one executable Python code block.
6. Avoid asking users to copy and paste results. Code should be self-contained and provide outputs directly.
7. If an error occurs, provide a corrected code block. Offer complete solutions rather than partial code snippets or modifications.
8. Verify solutions rigorously and ensure the code addresses the task effectively without user intervention beyond code execution.
Use this approach to ensure that the user receives precise, direct, and executable Python code for their tasks."#.to_string();

    // Reply "TERMINATE" in the end when everything is done.

    pub static ref FUNCTON_CALL_SYSTEM_PROMPT: String =
        r#"<|im_start|>system You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>
    
    The function `get_webpage_text` retrieves all text content from a given URL. For example, calling `get_webpage_text("https://example.com")` will fetch the text from Example.com.
    
    The function `search_bing` performs a search using Bing and returns the results. For example, `search_bing("latest AI research trends")` will return search results related to the latest trends in AI research. 
    {
        "name": "get_webpage_text",
        "description": ""Retrieves all text content from a specified website URL.",
        "parameters": {
            "url": {
                "type": "string",
                "description": "The URL of the website from which to fetch the text content__"
            }
        },
        "required": ["url"],
        "type": "object"
    }
    
    {
        "name": "search_bing",
        "description": "Conduct a search using the Bing search engine and return the results.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query to be executed on Bing__"
            }
        },
        "required": ["query"],
        "type": "object"
    }
    For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
    <tool_call>
    {"arguments": <args-dict>, "name": <function-name>}
    </tool_call>"#.to_string();

    pub static ref ROUTING_BY_TOOLCALL_PROMPT: String =
        r#"
    <|im_start|>system You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. Your sole responsibility is to route tasks to appropriate agents based on their capabilities using virtual toolcalls. Do not attempt to execute or handle any part of the user's task yourself. Here are the available tools: <tools>

    The virtual toolcall routing allows dispatching tasks based on agent capabilities:
    
    **coding_agent**: Specializes in generating clean, executable Python code for various tasks.
    **user_proxy**: Represents the user by delegating tasks to agents, reviewing their outputs, and ensuring tasks meet user requirements; it is also responsible for receiving final task results.
    
    Use "coding_agent" or "user_proxy" as virtual toolcall names with arguments specifying key points from input.
    
    For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
    <tool_call>
    {"arguments": <args-dict>, 
    "name":"<function-name>"}
    </tool_call>
    
    The following are examples of how to use these virtual toolcalls:
    
    1. **coding_agent**:
       - Description: Specializes in generating clean, executable Python code for various tasks.
       - Use this agent when you need specific Python code generated.
       - Example usage:
           {
               "name": "coding_agent",
               "description": "Routes task to coding_agent for generating Python code.",
               "parameters": {
                   "key_points": {
                       "type": "string",
                       "description": "key points from input that describes what kind of problem a coding_agent needs to solve with Python code."
                   }
               },
               "required": ["key_points"],
               "type": "object"
           }
    
    2. **user_proxy**:
       - Description: Represents the user by delegating tasks to agents, reviewing their outputs, and ensuring tasks meet user requirements; it is also responsible for receiving final task results.
       - Use this agent when you're not explicitly asked to use code to solve a problem.
       - Use this agent when you need someone else (e.g., another agent) involved in completing or reviewing a task.
       - Use this agent when you're given some facts without any explicit user intentions expressed; you're expected only pass on such information without additional processing or interpretation.
       - Example usage:
           {
            "name": "user_proxy",
            "description": "Routes task to user_proxy for delegation and review.",
            "parameters": {
                "key_points":{
                    "type": "string",
                    "description": "Review generated report & provide feedback."
                }
             }, 
             "required": ["key_points"], 
             "type": "object"
          }  
    
    Examples of routing decisions:
    
    - If input involves providing an answer/result directly without needing new code generation (e.g., factual statements like weather updates), route it directly through "user_proxy".
    - If input requires specific programming solutions (e.g., writing new functions or scripts), route it through "coding_agent"."#.to_string();

    pub static ref FURTER_TASK_BY_TOOLCALL_PROMPT: String =
        r#"<|im_start|>system You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>
    
    The function "start_coding" generates clean, executable Python code for various tasks based on the user input. For example, calling "start_coding("key_points": "Create a Python script that reads a CSV file and plots a graph")" will generate Python code that performs this task.
    
    The function "get_webpage_text" retrieves all text content from a given URL, which can be useful for extracting information from web pages or articles. For example, calling "get_webpage_text("https://example.com")" will fetch the text from Example.com.
    
    The function "search_bing" performs an internet search using Bing and returns relevant results based on the query provided by the user. This can be useful for finding up-to-date information on various topics. For example, "search_bing("latest AI research trends")" will return search results related to the latest trends in AI research.
    
    {
        "name": "get_webpage_text",
        "description": "Retrieves all text content from a specified website URL.",
        "parameters": {
            "url": {
                "type": "string",
                "description": "The URL of the website from which to fetch the text content"
            }
        },
        "required": ["url"],
        "type": "object"
    }
    
    {
        "name": "start_coding",
            "description": "Generates clean, executable Python code for various tasks",
            "parameters": {
                "key_points": {
                    "type": "string",
                    "description": "Key points from input that describes what kind of problem needs to be solved with Python code."
                }
            },
            "required": ["key_points"],
            "type": "object"
 }
 
 {
    "name": "search_bing",
    "description": "Conducts an internet search using Bing search engine and returns relevant results.",
    "parameters": { 
         "query": { 
             "type": "string", 
             "description": "The search query to be executed on Bing" 
          } 
     }, 
     "required": ["query"], 
     "type": "object"
 }

Examples of toolcalls for different scenarios and tools:
1. To retrieve webpage text:
<tool_call>
{"arguments":{"url":"https://example.com"}, 
"name":"get_webpage_text"}
</tool_call>

2. To generate Python code:
<tool_call>
{"arguments":{"key_points":"Create a Python script that reads data from an API and stores it in a database"}, 
"name":"start_coding"}
</tool_call>

3. To perform an internet search:
<tool_call>
{"arguments":{"query":"best practices in software development"}, 
"name":"search_bing"}
</tool_call>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, 
"name":"<function-name>"}
</tool_call>"#.to_string();

    pub static ref ITERATE_CODING_START_TEMPLATE: Arc<Mutex<FormatterFn>> = Arc::new(
        Mutex::new(Box::new(|args: &[&str]| { format!("Here is the task for you: {}", args[0]) }))
    );

    pub static ref ITERATE_CODING_INVALID_TEMPLATE: String =
        "Failed to create valid Python code".to_string();

    pub static ref ITERATE_CODING_SUCCESS_TEMPLATE: Arc<Mutex<FormatterFn>> = Arc::new(
        Mutex::new(
            Box::new(|args: &[&str]| {
                format!(
                    "Successfully executed the code below:\n{}\n producing the following result:\n{}",
                    args[0],
                    args[1]
                )
            })
        )
    );

    pub static ref ITERATE_CODING_INCORRECT_TEMPLATE: Arc<Mutex<FormatterFn>> = Arc::new(
        Mutex::new(
            Box::new(|args: &[&str]| {
                format!(
                    "Executed the code below:\n{}\n producing the following result, but the result is incorrect:\n{}",
                    args[0],
                    args[1]
                )
            })
        )
    );

    pub static ref ITERATE_CODING_FAIL_TEMPLATE: Arc<Mutex<FormatterFn>> = Arc::new(
        Mutex::new(
            Box::new(|args: &[&str]| {
                format!(
                    "Failed to execute the code:\n{}, got the following errors:\n{}",
                    args[0],
                    args[1]
                )
            })
        )
    );

    pub static ref ITERATE_CODING_HISTORY_TEMPLATE: Arc<Mutex<FormatterFn>> = Arc::new(
        Mutex::new(
            Box::new(|args: &[&str]| {
                format!(
                    "Reminder: you are working towards solving the following task: {}. Here is a summary of the code iterations and their results: {}\nNow let's retry: take care not to repeat previous errors! Try to adopt different approaches.",
                    args[0],
                    args[1]
                )
            })
        )
    );

    pub static ref ITERATE_CODE_RETRY_TEMPLATE: Arc<Mutex<FormatterFn>> = Arc::new(
        Mutex::new(
            Box::new(|args: &[&str]| {
                format!(
                    "Error: {}\nNow let's retry: take care not to repeat previous errors! Try to adopt different approaches.",
                    args[0]
                )
            })
        )
    );
}

/* pub async fn entry(inp: &str) {
    use tokio::select;

    let mut agent = ImmutableAgent::coding_agent(None, "");
    loop {
        agent = select! {
            agent = group.next_agent() => {
                agent
            },
            _ = signal::ctrl_c() => {
                std::process::exit(0);
            },
        };

        agent.();
    }
} */
