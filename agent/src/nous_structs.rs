use endpoints::{
    chat::{
        ChatCompletionObject,
        ChatCompletionRequest,
        ChatCompletionRequestMessage,
        ChatCompletionRole,
        ChatCompletionUserMessageContent,
    },
    common::Usage,
};
use llama_core::LlamaCoreError;
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
// use crate::llm_llama_local::chat_inner_async;

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct NousToolCall {
    pub name: String,
    pub arguments: Option<HashMap<String, String>>,
}

#[allow(non_snake_case)]
#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub enum NousContent {
    Text(String),
    NousToolCall(NousToolCall),
}
trait UsageClone {
    fn clone(&self) -> Self;
}

impl UsageClone for Usage {
    fn clone(&self) -> Self {
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NousResponseMessage {
    pub content: NousContent,
    pub role: ChatCompletionRole,
    pub usage: Usage,
}

impl Clone for NousResponseMessage {
    fn clone(&self) -> Self {
        Self {
            content: self.content.clone(),
            role: self.role.clone(),
            usage: self.usage.clone(), // Use the custom clone method here
        }
    }
}

impl Clone for NousToolCall {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            arguments: self.arguments.clone(),
        }
    }
}

impl Clone for NousContent {
    fn clone(&self) -> Self {
        match &*self {
            NousContent::Text(s) => NousContent::Text(s.clone()),
            NousContent::NousToolCall(t) => NousContent::NousToolCall(t.clone()),
        }
    }
}
impl NousResponseMessage {
    pub fn content_to_string(&self) -> String {
        match &self.content {
            NousContent::Text(text) => text.clone(),
            NousContent::NousToolCall(tool_call) =>
                format!(
                    "tool_call: {}, arguments: {}",
                    tool_call.name,
                    tool_call.arguments
                        .as_ref()
                        .unwrap()
                        .into_iter()
                        .map(|(arg, val)| format!("{:?}: {:?}", arg, val))
                        .collect::<Vec<String>>()
                        .join(", ")
                ),
        }
    }
}

fn extract_json_from_xml_like(xml_like_data: &str) -> Option<String> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    if xml_like_data.trim().starts_with(start_tag) && xml_like_data.trim().ends_with(end_tag) {
        let start_pos = start_tag.len();
        let end_pos = xml_like_data.len() - end_tag.len();
        Some(xml_like_data[start_pos..end_pos].trim().to_string())
    } else {
        None
    }
}

pub fn output_nous_response(res_obj: ChatCompletionObject) -> NousResponseMessage {
    let usage = res_obj.usage;
    let msg_obj = &res_obj.choices[0].message;
    let role = msg_obj.role.clone(); // Assuming role is clonable

    let data = &msg_obj.content;
    println!(" data: {:?}", data.clone());

    let mut res = NousContent::Text(data.to_owned());
    if let Some(json_str) = extract_json_from_xml_like(data) {
        if let Ok(tool_call) = serde_json::from_str::<NousToolCall>(&json_str) {
            res = NousContent::NousToolCall(tool_call);
        }
    }
    NousResponseMessage {
        content: res,
        role,
        usage,
    }
}

pub async fn chat_completions_partial(
    chat_request: &mut ChatCompletionRequest,
    user_input: &str
) -> Result<NousResponseMessage, LlamaCoreError> {
    let user_message = ChatCompletionRequestMessage::new_user_message(
        ChatCompletionUserMessageContent::Text(user_input.to_string()),
        None
    );

    chat_request.messages.push(user_message);

    let res = llama_core::chat::chat_completions(chat_request).await?;

    let content = output_nous_response(res);

    Ok(content)
}

pub async fn chat_completions_full(
    chat_request: &mut ChatCompletionRequest,
    system_prompt: &str,
    user_input: &str
) -> Result<NousResponseMessage, LlamaCoreError> {
    let system_message = ChatCompletionRequestMessage::new_system_message(system_prompt, None);

    chat_request.messages.push(system_message);
    let user_message = ChatCompletionRequestMessage::new_user_message(
        ChatCompletionUserMessageContent::Text(user_input.to_string()),
        None
    );

    chat_request.messages.push(user_message);

    let res = llama_core::chat::chat_completions(chat_request).await?;

    let content = output_nous_response(res);

    Ok(content)
}
