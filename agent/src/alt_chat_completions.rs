use chat_prompts::{ chat::{ BuildChatPrompt, ChatPrompt }, PromptTemplateType };

use std::collections::HashMap;
use once_cell::sync::OnceCell;
use std::sync::RwLock;
use endpoints::chat::{
    ChatCompletionRequestMessage,
    ChatCompletionUserMessageContent,
    ContentPart,
};
use endpoints::{
    chat::{
        ChatCompletionChunk,
        ChatCompletionChunkChoice,
        ChatCompletionChunkChoiceDelta,
        ChatCompletionObject,
        ChatCompletionObjectChoice,
        ChatCompletionObjectMessage,
        ChatCompletionRequest,
        ChatCompletionRole,
    },
    common::{ FinishReason, Usage },
};
use error::{ BackendError, LlamaCoreError };
use llama_core::{ error, running_mode, Graph, Metadata, RunningMode };
use serde_json::Value;
use std::{ sync::{ Arc, Mutex }, time::SystemTime };

const PLUGIN_VERSION: usize = 1;

pub(crate) static CHAT_GRAPHS: OnceCell<Mutex<HashMap<String, Graph>>> = OnceCell::new();
// key: model_name, value: Graph
pub(crate) static EMBEDDING_GRAPHS: OnceCell<Mutex<HashMap<String, Graph>>> = OnceCell::new();
// cache bytes for decoding utf8
pub(crate) static CACHED_UTF8_ENCODINGS: OnceCell<Mutex<Vec<u8>>> = OnceCell::new();
// running mode
pub(crate) static RUNNING_MODE: OnceCell<RwLock<RunningMode>> = OnceCell::new();

pub(crate) const MAX_BUFFER_SIZE: usize = (2usize).pow(14) * 15 + 128;
pub(crate) const OUTPUT_TENSOR: usize = 0;

pub fn update_metadata(&chat: ChatCompletionRequest) -> Result<(), LlamaCoreError> {
    // update metadata
    let config = match serde_json::to_string(&chat.metadata) {
        Ok(config) => config,
        Err(e) => {
            return Err(LlamaCoreError::Operation(format!(
                "Fail to serialize metadata to a JSON string. {}",
                e
            )));
        }
    };
    set_tensor_data_u8(chat, 1, config.as_bytes())
}

pub async fn chat_completions(
    chat_request: &mut ChatCompletionRequest
) -> Result<ChatCompletionObject, LlamaCoreError> {
    let running_mode = running_mode()?;
    if running_mode == RunningMode::Embeddings {
        return Err(
            LlamaCoreError::Operation(
                format!("The chat completion is not supported in the {running_mode} mode.")
            )
        );
    }

    let model_name = chat_request.model.clone();
    let id = match &chat_request.user {
        Some(id) => id.clone(),
        None => gen_chat_id(),
    };

    // update metadata
    let mut metadata = update_metadata(&chat_request)?;

    // build prompt
    let (prompt, avaible_completion_tokens) = build_prompt(
        model_name.as_ref(),
        chat_request
    ).map_err(|e| LlamaCoreError::Operation(format!("Failed to build prompt. {}", e)))?;

    if metadata.log_prompts {
        print_log_begin_separator("PROMPT", Some("*"), None);
        println!("\n{}", &prompt);
        print_log_end_separator(Some("*"), None);
    }

    // update metadata n_predict
    update_n_predict(chat_request, &mut metadata, avaible_completion_tokens).await?;

    // set prompt
    set_prompt(model_name.as_ref(), &prompt)?;

    // compute
    compute(model_name.as_ref(), id)
}

pub(crate) fn gen_chat_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

pub(crate) fn get_output_buffer(graph: &Graph, index: usize) -> Result<Vec<u8>, LlamaCoreError> {
    let mut output_buffer: Vec<u8> = Vec::with_capacity(MAX_BUFFER_SIZE);

    let output_size: usize = graph
        .get_output(index, &mut output_buffer)
        .map_err(|e| {
            LlamaCoreError::Backend(
                BackendError::GetOutput(format!("Fail to get plugin metadata. {msg}", msg = e))
            )
        })?;

    unsafe {
        output_buffer.set_len(output_size);
    }

    Ok(output_buffer)
}

pub(crate) fn get_output_buffer_single(
    graph: &Graph,
    index: usize
) -> Result<Vec<u8>, LlamaCoreError> {
    let mut output_buffer: Vec<u8> = Vec::with_capacity(MAX_BUFFER_SIZE);

    let output_size: usize = graph
        .get_output_single(index, &mut output_buffer)
        .map_err(|e| {
            LlamaCoreError::Backend(
                BackendError::GetOutput(format!("Fail to get plugin metadata. {msg}", msg = e))
            )
        })?;

    unsafe {
        output_buffer.set_len(output_size);
    }

    Ok(output_buffer)
}

pub(crate) fn set_tensor_data_u8(
    graph: &mut Graph,
    idx: usize,
    tensor_data: &[u8]
) -> Result<(), LlamaCoreError> {
    if graph.set_input(idx, wasmedge_wasi_nn::TensorType::U8, &[1], tensor_data).is_err() {
        return Err(LlamaCoreError::Operation(String::from("Fail to set input tensor")));
    }

    Ok(())
}

/// Get the token information from the graph.
pub(crate) fn get_token_info_by_graph(graph: &Graph) -> Result<TokenInfo, LlamaCoreError> {
    let output_buffer = get_output_buffer(graph, 1)?;
    let token_info: Value = match serde_json::from_slice(&output_buffer[..]) {
        Ok(token_info) => token_info,
        Err(e) => {
            return Err(
                LlamaCoreError::Operation(format!("Fail to deserialize token info: {msg}", msg = e))
            );
        }
    };

    let prompt_tokens = match token_info["input_tokens"].as_u64() {
        Some(prompt_tokens) => prompt_tokens,
        None => {
            return Err(
                LlamaCoreError::Operation(String::from("Fail to convert `input_tokens` to u64."))
            );
        }
    };
    let completion_tokens = match token_info["output_tokens"].as_u64() {
        Some(completion_tokens) => completion_tokens,
        None => {
            return Err(
                LlamaCoreError::Operation(String::from("Fail to convert `output_tokens` to u64."))
            );
        }
    };
    Ok(TokenInfo {
        prompt_tokens,
        completion_tokens,
    })
}

/// Get the token information from the graph by the model name.
pub(crate) fn get_token_info_by_graph_name(
    name: Option<&String>
) -> Result<TokenInfo, LlamaCoreError> {
    match name {
        Some(model_name) => {
            let chat_graphs = CHAT_GRAPHS.get().ok_or(
                LlamaCoreError::Operation(
                    String::from("Fail to get the underlying value of `CHAT_GRAPHS`.")
                )
            )?;
            let chat_graphs = chat_graphs
                .lock()
                .map_err(|e| {
                    LlamaCoreError::Operation(
                        format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e)
                    )
                })?;
            match chat_graphs.get(model_name) {
                Some(graph) => get_token_info_by_graph(graph),
                None =>
                    Err(
                        LlamaCoreError::Operation(
                            format!(
                                "The model `{}` does not exist in the chat graphs.",
                                &model_name
                            )
                        )
                    ),
            }
        }
        None => {
            let chat_graphs = CHAT_GRAPHS.get().ok_or(
                LlamaCoreError::Operation(
                    String::from("Fail to get the underlying value of `CHAT_GRAPHS`.")
                )
            )?;
            let chat_graphs = chat_graphs
                .lock()
                .map_err(|e| {
                    LlamaCoreError::Operation(
                        format!("Fail to acquire the lock of `CHAT_GRAPHS`. {}", e)
                    )
                })?;

            match chat_graphs.iter().next() {
                Some((_, graph)) => get_token_info_by_graph(graph),
                None =>
                    Err(
                        LlamaCoreError::Operation(
                            String::from("There is no model available in the chat graphs.")
                        )
                    ),
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct TokenInfo {
    pub(crate) prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
}

pub(crate) fn print_log_begin_separator(
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

fn build_prompt(
    model_name: Option<&String>,
    // template: &ChatPrompt,
    chat_request: &mut ChatCompletionRequest,
) -> Result<(String, u64), LlamaCoreError> {
    let metadata = get_metadata(model_name)?;
    let ctx_size = metadata.ctx_size as u64;
    let chat_prompt = ChatPrompt::from(metadata.prompt_template);

    // compute max prompt tokens
    let max_prompt_tokens = ctx_size * 4 / 5;

    loop {
        // build prompt
        let prompt = match chat_prompt.build(&mut chat_request.messages) {
            Ok(prompt) => prompt,
            Err(e) => {
                return Err(LlamaCoreError::Operation(format!(
                    "Fail to build chat prompts: {msg}",
                    msg = e
                )))
            }
        };

        // set prompt
        set_prompt(model_name, &prompt)?;

        // Retrieve the number of prompt tokens.
        let token_info = get_token_info_by_graph_name(model_name)?;

        match token_info.prompt_tokens > max_prompt_tokens {
            true => {
                match chat_request.messages[0].role() {
                    ChatCompletionRole::System => {
                        if chat_request.messages.len() >= 4 {
                            if chat_request.messages[1].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(1);
                            }
                            if chat_request.messages[1].role() == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(1);
                            }
                        } else if chat_request.messages.len() == 3
                            && chat_request.messages[1].role() == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(1);
                        } else {
                            return Ok((prompt, ctx_size - max_prompt_tokens));
                        }
                    }
                    ChatCompletionRole::User => {
                        if chat_request.messages.len() >= 3 {
                            if chat_request.messages[0].role() == ChatCompletionRole::User {
                                chat_request.messages.remove(0);
                            }
                            if chat_request.messages[0].role() == ChatCompletionRole::Assistant {
                                chat_request.messages.remove(0);
                            }
                        } else if chat_request.messages.len() == 2
                            && chat_request.messages[0].role() == ChatCompletionRole::User
                        {
                            chat_request.messages.remove(0);
                        } else {
                            return Ok((prompt, ctx_size - max_prompt_tokens));
                        }
                    }
                    _ => panic!("Found a unsupported chat message role!"),
                }

                continue;
            }
            false => return Ok((prompt, ctx_size - max_prompt_tokens)),
        }
    }
}

fn print_log_end_separator(ch: Option<&str>, len: Option<usize>) {
    let ch = ch.unwrap_or("-");
    let mut separator = "\n\n".to_string();
    separator.push_str(ch.repeat(len.unwrap_or(100)).as_str());
    separator.push('\n');
    println!("{}", separator);
}

async fn update_n_predict(
    chat_request: &ChatCompletionRequest,
    metadata: &mut Metadata,
    available_completion_tokens: u64,
) -> Result<(), LlamaCoreError> {
    let mut should_update = false;

    // check if necessary to update n_predict with max_tokens
    if let Some(max_tokens) = chat_request.max_tokens {
        let max_completion_tokens = match available_completion_tokens < max_tokens {
            true => available_completion_tokens,
            false => max_tokens,
        };

        // update n_predict
        metadata.n_predict = max_completion_tokens;

        if !should_update {
            should_update = true;
        }
    } else if metadata.n_predict > available_completion_tokens {
        // update n_predict
        metadata.n_predict = available_completion_tokens;

        if !should_update {
            should_update = true;
        }
    }

    if should_update {
        // update the target graph with the new metadata
        set_metadata(chat_request.model.as_ref(), metadata)?;
    }

    Ok(())
}

fn set_prompt(model_name: Option<&String>, prompt: impl AsRef<str>) -> Result<(), LlamaCoreError> {
    match model_name {
        Some(model_name) => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;
            match chat_graphs.get_mut(model_name) {
                Some(graph) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                None => Err(LlamaCoreError::Operation(format!(
                    "The model `{}` does not exist in the chat graphs.",
                    &model_name
                ))),
            }
        }
        None => {
            let chat_graphs = CHAT_GRAPHS
                .get()
                .ok_or(LlamaCoreError::Operation(String::from(
                    "Fail to get the underlying value of `CHAT_GRAPHS`.",
                )))?;
            let mut chat_graphs = chat_graphs.lock().map_err(|e| {
                LlamaCoreError::Operation(format!(
                    "Fail to acquire the lock of `CHAT_GRAPHS`. {}",
                    e
                ))
            })?;

            match chat_graphs.iter_mut().next() {
                Some((_, graph)) => {
                    let tensor_data = prompt.as_ref().as_bytes().to_vec();
                    set_tensor_data_u8(graph, 0, &tensor_data)
                }
                None => Err(LlamaCoreError::Operation(String::from(
                    "There is no model available in the chat graphs.",
                ))),
            }
        }
    }
}
use core::error::Error;
pub fn compute(&mut self) -> Result<(), Error> {
    syscall::compute(self.ctx_handle)
}