use anyhow;
use regex::Regex;
// use rustpython::vm::Settings;
use rustpython::vm;
use rustpython::InterpreterConfig;

pub async fn run_python_wrapper(code_wrapped_in_text: &str) -> (bool, String, String) {
    let code = extract_code(code_wrapped_in_text);

    match run_python_capture(&code) {
        Ok(success_result_text) => (true, code, success_result_text),

        Err(err_msg) => (false, code, err_msg),
    }
}

pub fn run_python_capture(code: &str) -> anyhow::Result<String, String> {
    let interpreter = InterpreterConfig::new().init_stdlib().interpreter();
    interpreter.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        let code_with_redirect_and_output =
            format!("import io\nimport sys\noutput = io.StringIO()\nsys.stdout = output\n{}\ncaptured_output = output.getvalue()", code);

        let code_obj = vm
            .compile(
                &code_with_redirect_and_output,
                vm::compiler::Mode::Exec,
                "<embedded>".to_owned()
            )
            .map_err(|err| format!("Compilation error: {}", err))?;

        match vm.run_code_obj(code_obj, scope.clone()) {
            Ok(_) => {
                match scope.globals.get_item("captured_output", vm) {
                    Ok(res) =>
                        match res.downcast_ref::<vm::builtins::PyStr>() {
                            Some(py_str) => Ok(py_str.as_str().to_string()),
                            None => Err("res is not a string".to_string()),
                        }
                    Err(_) => Err("error getting captured_output".to_string()),
                }
            }
            Err(e) => {
                let error_message = if let Some(args) = e.args().as_slice().first() {
                    args.downcast_ref::<vm::builtins::PyStr>()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "Unknown error".to_string())
                } else {
                    "No error message available".to_string()
                };
                Err(format!("Code execution error message: {}", error_message))
            }
        }
    })
}

pub fn extract_code(text: &str) -> String {
    let multi_line_pattern = r"(?s)```python(.*?)```";
    let mut program = String::new();

    let multi_line_regex = Regex::new(multi_line_pattern).unwrap();
    for cap in multi_line_regex.captures_iter(text) {
        if let Some(code) = cap.get(1) {
            program.push_str(code.as_str().trim());
        }
    }

    program
}

pub fn extract_code_blocks(
    text: &str,
    detect_single_line_code: bool
) -> Vec<(Option<String>, String)> {
    // Adjust regex pattern to handle both Unix and Windows line endings and optional language specifier
    let multi_line_pattern = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```";
    let single_line_pattern = r"`([^`]+)`";
    let mut results: Vec<(Option<String>, String)> = Vec::new();

    let multi_line_regex = Regex::new(multi_line_pattern).unwrap();
    for cap in multi_line_regex.captures_iter(text) {
        let language = cap.get(1).map_or(None, |m| Some(m.as_str().trim().to_string()));
        let code = cap.get(2).unwrap().as_str().trim().to_string();
        results.push((language.clone(), code.clone()));
        // println!("Matched multi-line code block: Language: {:?}, Code: {}", language, code);
    }

    if detect_single_line_code {
        let single_line_regex = Regex::new(single_line_pattern).unwrap();
        for cap in single_line_regex.captures_iter(text) {
            results.push((None, cap.get(1).unwrap().as_str().trim().to_string()));
            // println!("Matched single-line code: {}", cap.get(1).unwrap().as_str().trim());
        }
    }

    results
}

// export DYLD_LIBRARY_PATH=/Users/jichen/miniconda3/lib:$DYLD_LIBRARY_PATH
// export PYO3_PYTHON=/Users/jichen/miniconda3/bin/python
// export DYLD_LIBRARY_PATH=/Users/jichen/miniconda3/lib
