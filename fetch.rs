use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use futures_util::StreamExt;

#[derive(Serialize)]
struct Prompt {
    model: String,
    prompt: String,
}

#[derive(Deserialize, Debug)]
struct ResponseChunk {
    response: String,
    done: bool,
    // Add more fields if needed
}

#[derive(Debug, Deserialize)]
struct StructuredInsight {
    topic: Option<String>,
    concept: Option<String>,
    definition: Option<String>,
    example: Option<String>,
    subtopic: Option<String>,
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
struct Concept {
    definition: Option<String>,
    examples: HashSet<String>,
    related_concepts: HashSet<String>,
    subtopics: HashSet<String>,
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
        let concept_entry = self.concepts.entry(concept.to_string()).or_default();
        concept_entry.related_concepts.insert(related);
    }

    fn add_definition(&mut self, concept: String, definition: String) {
        let concept_entry = self.concepts.entry(concept).or_default();
        concept_entry.definition = Some(definition);
    }

    fn add_example(&mut self, concept: &str, example: String) {
        let concept_entry = self.concepts.entry(concept.to_string()).or_default();
        concept_entry.examples.insert(example);
    }

    fn add_subtopic(&mut self, concept: &str, subtopic: String) {
        let concept_entry = self.concepts.entry(concept.to_string()).or_default();
        concept_entry.subtopics.insert(subtopic);
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

    let prompt_text = format!("How does {} relate to other fields of science?", input.trim());

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
            let line = buffer[..pos].trim().to_string(); // clone to avoid borrow issue
            buffer = buffer[pos + 1..].to_string();      // now safe to mutate

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
    println!("Summary: {}", text);

    extract_insights(&text, knowledge, client).await?;

    let mut previous_concepts = knowledge.concepts.clone();
    let mut depth = 1;
    loop {
        if depth == 0{
            break
        }
        let general_subtopics = knowledge
            .concepts
            .get("General")
            .map(|c| c.subtopics.clone())
            .unwrap_or_default();

        if general_subtopics.is_empty() {
            println!("No more subtopics to explore.");
            break;
        }

        let subtopic_str = general_subtopics.iter().cloned().collect::<Vec<_>>().join(", ");

        let prompt_str = format!(
            "Please provide detailed explanations of the following subtopics: {}. Describe how each relates to {}.",
            subtopic_str, initial_prompt
        );

        let prompt = Prompt {
            model: MODEL.to_string(),
            prompt: prompt_str,
        };

        let response = client
            .post(OLLAMA_API_URL)
            .json(&prompt)
            .send()
            .await?;

        let text = get_streamed_text(response).await?;
        println!("Summary: {}", text);

        extract_insights(&text, knowledge, client).await?;

        if previous_concepts == knowledge.concepts {
            println!("No new concepts found, stopping.");
            break;
        }

        previous_concepts = knowledge.concepts.clone();

        depth -= 1; //temporary DEBUG
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
        "Analyze the following text and return JSON with these fields: topic, concept, definition, example, subtopic.\n\
        Example JSON format:\n\
        {{ \"topic\": \"Physics\", \"concept\": \"Gravity\", \"definition\": \"A force...\", \"example\": \"An apple falling...\", \"subtopic\": \"Newton's Laws\" }}\n\n\
        Test: \"{}\"",
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

    // Try parsing the model's response as JSON

    match extract_json_block(&raw_text)
    .and_then(|json| serde_json::from_str::<Vec<StructuredInsight>>(&json).ok())
    {
        Some(insights) => {
            for insight in insights {
                if let Some(concept) = &insight.concept {
                    knowledge.add_concept(concept.clone());

                    if let Some(topic) = &insight.topic {
                        knowledge.add_related_concept(concept, topic.clone());
                    }

                    if let Some(def) = &insight.definition {
                        knowledge.add_definition(concept.clone(), def.clone());
                    }

                    if let Some(ex) = &insight.example {
                        knowledge.add_example(concept, ex.clone());
                    }
                }

                if let Some(sub) = &insight.subtopic {
                    if !sub.is_empty() {
                        knowledge.add_subtopic("General", sub.clone());
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

        if !details.subtopics.is_empty() {
            writeln!(file, "  Subtopics:").unwrap();
            for st in &details.subtopics {
                writeln!(file, "    - {}", st).unwrap();
            }
        }

        writeln!(file).unwrap();
    }
}