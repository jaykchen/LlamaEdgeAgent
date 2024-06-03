use chat_prompts::PromptTemplateType;
use clap::Parser;
use llama_agent::immutable_agent::*;
use endpoints::chat::{
    ChatCompletionRequestBuilder,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSampling,
};
use llama_core::{ init_core_context, MetadataBuilder };
use serde::{ Deserialize, Serialize };

#[derive(Debug, Parser)]
#[command(author, about, version, long_about = None)]
struct Cli {
    /// Model name
    #[arg(short, long, default_value = "default")]
    model_name: String,
    /// Model alias
    #[arg(short = 'a', long, default_value = "default")]
    model_alias: String,
    /// Size of the prompt context
    #[arg(short, long, default_value = "512")]
    ctx_size: u64,
    /// Number of tokens to predict
    #[arg(short, long, default_value = "1024")]
    n_predict: u64,
    /// Number of layers to run on the GPU
    #[arg(short = 'g', long, default_value = "100")]
    n_gpu_layers: u64,
    /// Batch size for prompt processing
    #[arg(short, long, default_value = "512")]
    batch_size: u64,
    /// Temperature for sampling
    #[arg(long, conflicts_with = "top_p")]
    temp: Option<f64>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 1.0 = disabled
    #[arg(long, conflicts_with = "temp")]
    top_p: Option<f64>,
    /// Penalize repeat sequence of tokens
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f64,
    /// Repeat alpha presence penalty. 0.0 = disabled
    #[arg(long, default_value = "0.0")]
    presence_penalty: f64,
    /// Repeat alpha frequency penalty. 0.0 = disabled
    #[arg(long, default_value = "0.0")]
    frequency_penalty: f64,
    /// Sets the prompt template.
    #[arg(short, long, value_parser = clap::value_parser!(PromptTemplateType), required = true)]
    prompt_template: PromptTemplateType,
    /// Halt generation at PROMPT, return control.
    #[arg(short, long)]
    reverse_prompt: Option<String>,
    /// System prompt message string.
    #[arg(short, long)]
    system_prompt: Option<String>,
    /// Print prompt strings to stdout
    #[arg(long)]
    log_prompts: bool,
    /// Print statistics to stdout
    #[arg(long)]
    log_stat: bool,
    /// Print all log information to stdout
    #[arg(long)]
    log_all: bool,
    /// enable streaming stdout
    #[arg(long, default_value = "false")]
    disable_stream: bool,
}

#[allow(unreachable_code)]
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // get the environment variable `PLUGIN_DEBUG`
    let plugin_debug = std::env::var("PLUGIN_DEBUG").unwrap_or_default();
    let plugin_debug = match plugin_debug.is_empty() {
        true => false,
        false => plugin_debug.to_lowercase().parse::<bool>().unwrap_or(false),
    };

    // parse the command line arguments
    let cli = Cli::parse();

    // log version
    log(format!("\n[INFO] llama-chat version: {}", env!("CARGO_PKG_VERSION")));

    // log the cli options
    log(format!("[INFO] Model name: {}", &cli.model_name));
    log(format!("[INFO] Model alias: {}", &cli.model_alias));
    log(format!("[INFO] Prompt template: {}", &cli.prompt_template));

    // reverse prompt
    if let Some(reverse_prompt) = &cli.reverse_prompt {
        log(format!("[INFO] reverse prompt: {}", reverse_prompt));
    }
    // system prompt
    if let Some(system_prompt) = &cli.system_prompt {
        log(format!("[INFO] system prompt: {}", system_prompt));
    }

    // create a MetadataBuilder instance
    let builder = MetadataBuilder::new(&cli.model_name, &cli.model_alias, cli.prompt_template)
        .with_ctx_size(8192)
        .with_n_predict(999)
        .with_n_gpu_layers(cli.n_gpu_layers)
        .with_batch_size(cli.batch_size)
        .with_repeat_penalty(cli.repeat_penalty)
        .with_presence_penalty(cli.presence_penalty)
        .with_frequency_penalty(cli.frequency_penalty)
        .with_reverse_prompt(cli.reverse_prompt)
        .enable_prompts_log(cli.log_prompts || cli.log_all)
        .enable_plugin_log(cli.log_stat || cli.log_all)
        .enable_debug_log(plugin_debug)
        .with_temperature(0.1)
        .with_top_p(1.0);
    // create a Metadata instance
    let metadata = builder.build();

    // initialize the core context
    init_core_context(Some(&[metadata]), None)?;

    // get the plugin version info
    let plugin_info = llama_core::get_plugin_info()?;
    log(
        format!(
            "[INFO] Wasi-nn-ggml plugin: b{build_number} (commit {commit_id})",
            build_number = plugin_info.build_number,
            commit_id = plugin_info.commit_id
        )
    );

    // create a ChatCompletionRequestSampling instance
    let sampling = if cli.temp.is_none() && cli.top_p.is_none() {
        ChatCompletionRequestSampling::Temperature(1.0)
    } else if let Some(temp) = cli.temp {
        ChatCompletionRequestSampling::Temperature(temp)
    } else if let Some(top_p) = cli.top_p {
        ChatCompletionRequestSampling::TopP(top_p)
    } else {
        let temp = cli.temp.unwrap();
        ChatCompletionRequestSampling::Temperature(temp)
    };

    // create a chat request
    let mut chat_request = ChatCompletionRequestBuilder::new(&cli.model_name, vec![])
        .with_presence_penalty(cli.presence_penalty)
        .with_frequency_penalty(cli.frequency_penalty)
        .with_sampling(sampling)
        .enable_stream(!cli.disable_stream)
        .build();

    // add system message if provided
    if let Some(system_prompt) = &cli.system_prompt {
        let system_message = ChatCompletionRequestMessage::new_system_message(system_prompt, None);

        chat_request.messages.push(system_message);
    }

    let readme =
        "
================================== Running in interactive mode. ===================================\n
    - Press [Ctrl+C] to interject at any time.
    - Press [Return] to end the input.
    - For multi-line inputs, end each line with '\\' and press [Return] to get another line.\n";
    log(readme);

    let user_proxy = ImmutableAgent::simple("user_proxy", "you're user_proxy");

    loop {
        println!("\n[You]: ");
        let user_input = read_input();

        // put the user message into the messages sequence of chat_request
        // let user_message = ChatCompletionRequestMessage::new_user_message(
        //     ChatCompletionUserMessageContent::Text(user_input),
        //     None
        // );

        // chat_request.messages.push(user_message);

        if cli.log_stat || cli.log_all {
            print_log_begin_separator("STATISTICS (Set Input)", Some("*"), None);
        }

        if cli.log_stat || cli.log_all {
            print_log_end_separator(Some("*"), None);
        }

        println!("\n[Bot]:");

        let task_vec = user_proxy.planning(&mut chat_request, &user_input).await;

        let o = user_proxy.stepper(&mut chat_request, &task_vec).await;

        println!("{:?}", o);

        // let mut assistant_answer = String::new();

        // let res = chat_completions_full(
        //     &mut chat_request,
        //     "You're a joyful assisant",
        //     &user_input
        // ).await?;

        // let assistant_answer = res.content_to_string();

        // match llama_core::chat::chat_completions(&mut chat_request).await {
        //     Ok(chunk) => {
        //         let content = &chunk.choices[0].message.content;
        //         if content.is_empty() {
        //             continue;
        //         }
        //         if assistant_answer.is_empty() {
        //             let content = content.trim_start();
        //             print!("{}", content);
        //             assistant_answer.push_str(content);
        //         } else {
        //             print!("{content}");
        //             assistant_answer.push_str(content);
        //         }
        //         io::stdout().flush().unwrap();
        //         println!();
        //     }
        //     Err(e) => bail!("Fail to generate completion. Reason: {msg}", msg = e),
        // }

        // let assistant_message = ChatCompletionRequestMessage::new_assistant_message(
        //     Some(assistant_answer),
        //     None,
        //     None
        // );
        // chat_request.messages.push(assistant_message);
    }

    Ok(())
}

// For single line input, just press [Return] to end the input.
// For multi-line input, end your input with '\\' and press [Return].
//
// For example:
//  [You]:
//  what is the capital of France?[Return]
//
//  [You]:
//  Count the words in the following sentence: \[Return]
//  \[Return]
//  You can use Git to save new files and any changes to already existing files as a bundle of changes called a commit, which can be thought of as a “revision” to your project.[Return]
//
fn read_input() -> String {
    let mut answer = String::new();
    loop {
        let mut temp = String::new();
        std::io::stdin().read_line(&mut temp).expect("The read bytes are not valid UTF-8");

        if temp.ends_with("\\\n") {
            temp.pop();
            temp.pop();
            temp.push('\n');
            answer.push_str(&temp);
            continue;
        } else if temp.ends_with('\n') {
            answer.push_str(&temp);
            return answer;
        } else {
            return answer;
        }
    }
}

fn print_log_begin_separator(
    title: impl AsRef<str>,
    ch: Option<&str>,
    len: Option<usize>
) -> usize {
    let title = format!(" [LOG: {}] ", title.as_ref());

    let total_len: usize = len.unwrap_or(100);
    let separator_len: usize = (total_len - title.len()) / 2;

    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push_str(&title);
    separator.push_str(ch.repeat(separator_len).as_str());
    separator.push('\n');
    println!("{}", separator);
    total_len
}

fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push('\n');
    println!("{}", separator);
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct Metadata {
    // * Plugin parameters (used by this plugin):
    #[serde(rename = "enable-log")]
    pub log_enable: bool,
    // #[serde(rename = "enable-debug-log")]
    // pub debug_log: bool,
    // #[serde(rename = "stream-stdout")]
    // pub stream_stdout: bool,
    #[serde(rename = "embedding")]
    pub embeddings: bool,
    #[serde(rename = "n-predict")]
    pub n_predict: u64,
    #[serde(skip_serializing_if = "Option::is_none", rename = "reverse-prompt")]
    pub reverse_prompt: Option<String>,
    // pub mmproj: String,
    // pub image: String,

    // * Model parameters (need to reload the model if updated):
    #[serde(rename = "n-gpu-layers")]
    pub n_gpu_layers: u64,
    // #[serde(rename = "main-gpu")]
    // pub main_gpu: u64,
    // #[serde(rename = "tensor-split")]
    // pub tensor_split: String,

    // * Context parameters (used by the llama context):
    #[serde(rename = "ctx-size")]
    pub ctx_size: u64,
    #[serde(rename = "batch-size")]
    pub batch_size: u64,

    // * Sampling parameters (used by the llama sampling context).
    #[serde(rename = "temp")]
    pub temperature: f64,
    #[serde(rename = "top-p")]
    pub top_p: f64,
    #[serde(rename = "repeat-penalty")]
    pub repeat_penalty: f64,
    #[serde(rename = "presence-penalty")]
    pub presence_penalty: f64,
    #[serde(rename = "frequency-penalty")]
    pub frequency_penalty: f64,
}

fn log(msg: impl std::fmt::Display) {
    println!("{}", msg);
}
