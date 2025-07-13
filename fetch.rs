use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use futures_util::StreamExt;
use tokio;
use tokio::time::{sleep, Duration};

#[derive(Serialize)]
struct Prompt {
    model: String,
    prompt: String,
}

#[derive(Deserialize, Debug)]
struct ResponseChunk {
    response: String,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct StructuredInsight {
    topic: Option<String>,
    concept: Option<String>,
    definition: Option<String>,
    example: Option<String>,
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
struct Concept {
    definition: Option<String>,
    examples: HashSet<String>,
    related_concepts: HashSet<String>,
}

#[derive(Default, Clone, Debug)]
struct Knowledge {
    concepts: HashMap<String, Concept>,
}

impl Knowledge {
    fn add_concept(&mut self, concept: String) {
        self.concepts.entry(concept).or_default();
    }

    fn add_related_concept(&mut self, concept: &str, related: String) {
        self.concepts
            .entry(concept.to_string())
            .or_default()
            .related_concepts
            .insert(related);
    }

    fn add_definition(&mut self, concept: String, definition: String) {
        self.concepts.entry(concept).or_default().definition = Some(definition);
    }

    fn add_example(&mut self, concept: &str, example: String) {
        self.concepts
            .entry(concept.to_string())
            .or_default()
            .examples
            .insert(example);
    }
}

const OLLAMA_API_URL: &str = "http://192.168.1.151/api/generate";
const MODEL: &str = "llama3.1:8b";

#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {
    let client = Client::new();

    print!("What science field(s) are you trying to document? --> ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");

    let prompt_text = format!(
        "How does {} relate to other fields of science?",
        input.trim()
    );

    let mut knowledge = Knowledge::default();

    build_documentation(&mut knowledge, &client, prompt_text).await?;
    write_documentation_to_file(&knowledge);

    Ok(())
}

async fn get_streamed_text(response: reqwest::Response) -> Result<String, reqwest::Error> {
    let mut full_text = String::new();
    let mut buffer = String::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            match serde_json::from_str::<ResponseChunk>(&line) {
                Ok(json_chunk) => {
                    full_text.push_str(&json_chunk.response);
                    if json_chunk.done {
                        return Ok(full_text);
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to parse line as JSON: {}\nError: {}", line, e);
                }
            }
        }
    }

    Ok(full_text)
}

async fn build_documentation(
    knowledge: &mut Knowledge,
    client: &Client,
    initial_prompt: String,
) -> Result<(), reqwest::Error> {
    knowledge.add_concept("General".to_string());

    // Start with the initial prompt
    let prompt = Prompt {
        model: MODEL.to_string(),
        prompt: initial_prompt.clone(),
    };

    let response = client
        .post(OLLAMA_API_URL)
        .json(&prompt)
        .send()
        .await?;

    let text = get_streamed_text(response).await?;
    println!("Initial Summary: {}", text);

    extract_insights(&text, knowledge, client).await?;

    // Recursive exploration
    let mut explored: HashSet<String> = HashSet::new();
    let mut to_explore: Vec<String> = knowledge
    .concepts
    .keys()
    .filter(|c| *c != "General") // Skip "General"
    .cloned()
    .collect();

    while let Some(concept) = to_explore.pop() {
        if explored.contains(&concept) {
            continue;
        }

        explored.insert(concept.clone());

        let knowledge_depth = to_explore.len();

        println!("30 seconds CoolDown starts...");
        sleep(Duration::from_secs(30)).await;
        println!("Remaining concepts to explore: {}", knowledge_depth);
        println!("Exploring related concept: {}", concept);

        let prompt_text = format!(
            "In the context of {}, how does the concept '{}' relate to other scientific disciplines or subfields? List related concepts, define them, and provide examples.",
            initial_prompt, concept
        );

        let prompt = Prompt {
            model: MODEL.to_string(),
            prompt: prompt_text,
        };

        let response = client
            .post(OLLAMA_API_URL)
            .json(&prompt)
            .send()
            .await?;

        let text = get_streamed_text(response).await?;
        println!("Summary for '{}': {}", concept, text);

        extract_insights(&text, knowledge, client).await?;

        if let Some(concept_entry) = knowledge.concepts.get(&concept) {
            for related in &concept_entry.related_concepts {
                if !explored.contains(related) && !to_explore.contains(related) {
                    to_explore.push(related.clone());
                }
            }
        }
    }

    Ok(())
}


fn extract_json_block(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    Some(text[start..=end].to_string())
}

async fn extract_insights(
    text: &str,
    knowledge: &mut Knowledge,
    client: &Client,
) -> Result<(), reqwest::Error> {
    let prompt = format!(
        "Analyze the following text and return JSON with these fields: topic, concept, definition, example.\n\
        Example JSON format:\n\
        {{ \"topic\": \"Physics\", \"concept\": \"Gravity\", \"definition\": \"A force...\", \"example\": \"An apple falling...\" }}\n\n\
        Text: \"{}\"",
        text
    );

    let request_body = Prompt {
        model: MODEL.to_string(),
        prompt,
    };

    let response = client
        .post(OLLAMA_API_URL)
        .json(&request_body)
        .send()
        .await?;

    let raw_text = get_streamed_text(response).await?;
    println!("Raw model output:\n{}", raw_text);

    match extract_json_block(&raw_text)
        .and_then(|json| serde_json::from_str::<Vec<StructuredInsight>>(&json).ok())
    {
        Some(insights) => {
            for insight in insights {
                if let Some(concept) = &insight.concept {
                    knowledge.add_concept(concept.clone());

                    if let Some(topic) = &insight.topic {
                        knowledge.add_related_concept(concept, topic.clone());
                        knowledge.add_related_concept(topic, concept.clone()); // Add reverse link
                    }

                    if let Some(def) = &insight.definition {
                        knowledge.add_definition(concept.clone(), def.clone());
                    }

                    if let Some(ex) = &insight.example {
                        knowledge.add_example(concept, ex.clone());
                    }
                }
            }
        }
        None => {
            eprintln!("Failed to parse JSON array or extract it:\n{}", raw_text);
        }
    }

    Ok(())
}

fn write_documentation_to_file(knowledge: &Knowledge) {
    use std::io::Write;
    let mut file = std::fs::File::create("documentation.txt").expect("Failed to create file");

    for (concept, details) in &knowledge.concepts {
        writeln!(file, "Concept: {}", concept).unwrap();

        if let Some(def) = &details.definition {
            writeln!(file, "  Definition: {}", def).unwrap();
        }

        if !details.examples.is_empty() {
            writeln!(file, "  Examples:").unwrap();
            for example in &details.examples {
                writeln!(file, "    - {}", example).unwrap();
            }
        }

        if !details.related_concepts.is_empty() {
            writeln!(file, "  Related Concepts:").unwrap();
            for rc in &details.related_concepts {
                writeln!(file, "    - {}", rc).unwrap();
            }
        }

        writeln!(file).unwrap();
    }
}
