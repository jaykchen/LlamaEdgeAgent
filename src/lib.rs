// pub mod exec_python;
// use exec_python::run_python_capture;
// pub mod llama_structs;
// pub mod llm_llama_local;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type FormatterFn = Box<dyn (Fn(&[&str]) -> String) + Send + Sync>;

// #[no_mangle]
// #[tokio::main(flavor = "current_thread")]
// pub async fn on_deploy() {
//     create_endpoint().await;
// }

// #[request_handler(get)]
// async fn handler(
//     _headers: Vec<(String, String)>,
//     _subpath: String,
//     _qry: HashMap<String, Value>,
//     _body: Vec<u8>,
// ) {
//     dotenv().ok();
//     let mut router = Router::new();
//     router
//         .insert("/run", vec![post(run_code_by_post_handler)])
//         .unwrap();
//     if let Err(e) = route(router).await {
//         match e {
//             RouteError::NotFound => {
//                 send_response(404, vec![], b"No route matched".to_vec());
//             }
//             RouteError::MethodNotAllowed => {
//                 send_response(405, vec![], b"Method not allowed".to_vec());
//             }
//         }
//     }
// }

async fn _run_code_by_post_handler(
    _headers: Vec<(String, String)>,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>,
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
            return;
        }
    };
    if let Some(code) = load.code {
        todo!();
        // log::info!("{:?}", res.clone());
        // send_response(
        //     200,
        //     vec![
        //         (String::from("content-type"), String::from("application/json")),
        //         (String::from("Access-Control-Allow-Origin"), String::from("*"))
        //     ],
        //     json!(res).to_string().as_bytes().to_vec()
        // );
    }
}
