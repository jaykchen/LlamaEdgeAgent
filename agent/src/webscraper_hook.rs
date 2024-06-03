use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, USER_AGENT};
use reqwest::Client;
use serde::Deserialize;

pub async fn get_webpage_text(url: String) -> anyhow::Result<String> {
    let client = Client::builder().build()?;

    let url = format!("https://code.flows.network/lambda/nsdNiGHUlT?url={url}");

    let res = client.get(&url).send().await?.text().await?;
    Ok(res)
}

pub async fn search_with_bing(query: &str) -> anyhow::Result<String> {
    #[allow(unused)]
    #[allow(non_snake_case)]
    #[derive(Debug, Clone, Deserialize)]
    struct QueryContext {
        originalQuery: String,
    }

    #[allow(unused)]
    #[allow(non_snake_case)]
    #[derive(Debug, Clone, Deserialize)]
    struct WebPage {
        id: String,
        name: String,
        url: String,
        isFamilyFriendly: bool,
        displayUrl: String,
        snippet: String,
        dateLastCrawled: String,
        language: String,
        isNavigational: bool,
    }

    #[allow(unused)]
    #[allow(non_snake_case)]
    #[derive(Debug, Clone, Deserialize)]
    struct WebPages {
        webSearchUrl: String,
        totalEstimatedMatches: u64,
        value: Vec<WebPage>,
    }

    #[allow(unused)]
    #[derive(Debug, Clone, Deserialize)]
    struct RankingResponse {
        mainline: Mainline,
    }

    #[allow(unused)]
    #[derive(Debug, Clone, Deserialize)]
    struct Mainline {
        items: Vec<Item>,
    }

    #[allow(unused)]
    #[allow(non_snake_case)]
    #[derive(Debug, Clone, Deserialize)]
    struct Item {
        answerType: String,
        resultIndex: u64,
        value: ItemValue,
    }

    #[allow(unused)]
    #[derive(Debug, Clone, Deserialize)]
    struct ItemValue {
        id: String,
    }

    #[allow(unused)]
    #[allow(non_snake_case)]
    #[derive(Debug, Clone, Deserialize)]
    struct SearchResponse {
        _type: String,
        queryContext: QueryContext,
        webPages: WebPages,
        rankingResponse: RankingResponse,
    }

    let encoded_query = urlencoding::encode(query);

    let url_str =
        format!("https://api.bing.microsoft.com/v7.0/search?count=1&q={}&responseFilter=Webpages&setLang=en", encoded_query);
    let mut headers = HeaderMap::new();
    let bing_key = std::env::var("BING_API_KEY").unwrap_or("bing_api_key not found".to_string());
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));

    headers.insert(
        "Ocp-Apim-Subscription-Key",
        HeaderValue::from_str(&bing_key)?,
    );
    let client = Client::builder().default_headers(headers).build()?;

    let res = client.get(&url_str).send().await?.text().await?;

    let search_response = serde_json::from_slice::<SearchResponse>(res.as_bytes())?;
    let out = search_response
        .webPages
        .value
        .iter()
        .map(|val| format!("webpage at {} states: {}", val.url, val.snippet))
        .collect::<Vec<String>>()
        .join("\n");
    Ok(out)
}
