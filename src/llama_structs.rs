use llama_core::{
    CompletionRequest,
    ChatCompletionRequestMessage,
    CompletionUsage,
    CompletionObject,
    CompletionChoice,
    CompletionRequestBuilder,
    FinishReason,
    Usage,
    ChatCompletionObjectMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
    ToolChoiceTool,
    ChatCompletionUserMessageContent,
    Fuction,
    ToolCall,
    ChatCompletionToolMessage,
    ChatCompletionObject,
    JSONSchemaDefine,
    ChatCompletionRequestFunctionParameters,
    ChatCompletionRequestFunction,
    TextContentPart,
    ChatCompletionUserMessageContent,
};

use serde::{ Deserialize, Serialize };
use std::collections::HashMap;

use crate::llm_llama_local::chat_inner_async;


#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct LlamaResponseMessage {
    pub content: Content,
    pub role: Role,
    pub usage: CompletionUsage,
}
impl LlamaResponseMessage {
    pub fn content_to_string(&self) -> String {
        match &self.content {
            Content::Text(text) => text.clone(),
            Content::ToolCall(tool_call) =>
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

pub fn output_llama_response(
    res_obj: CreateChatCompletionResponse
) -> Option<LlamaResponseMessage> {
    let usage = res_obj.clone().usage.unwrap();
    let msg_obj = res_obj.clone().choices[0].message.clone();
    let role = msg_obj.clone().role;
    if let Some(data) = msg_obj.content {
        if let Some(json_str) = extract_json_from_xml_like(&data) {
            println!("{:?}", json_str.clone());
            let tool_call: ToolCall = serde_json::from_str(&json_str).unwrap();
            return Some(LlamaResponseMessage {
                content: Content::ToolCall(tool_call),
                role: role,
                usage: usage,
            });
        } else {
            return Some(LlamaResponseMessage {
                content: Content::Text(data.to_owned()),
                role: role,
                usage: usage,
            });
        }
    }
    None
}
