import { Ollama } from "@langchain/ollama";

import { Document } from "@langchain/core/documents";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/ollama";
import * as fs from 'fs';
import * as path from 'path';
import { MarkdownTextSplitter } from "langchain/text_splitter";

// Configuration
const OLLAMA_BASE_URL = "http://localhost:11434";
const MODEL_NAME = "llama3.2:latest"; // Use the model you have in Ollama
const EMBED_MODEL = "granite-embedding:278m";
const COLLECTION_NAME = "alice_wonderland_book";
const MARKDOWN_DIR = "./book"; // Directory containing markdown files
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;

// Initialize Ollama
const ollama = new Ollama({
  baseUrl: OLLAMA_BASE_URL,
  model: MODEL_NAME,
});

// Initialize embeddings
const embeddings = new OllamaEmbeddings({
  baseUrl: OLLAMA_BASE_URL,
  model: EMBED_MODEL,
});

async function getDataVectorStor() {
  const vectorStore = await Chroma.fromExistingCollection(embeddings,{
    collectionName: COLLECTION_NAME,
    url: "http://localhost:8000",
  });

  return vectorStore;
}

// Create RAG pipeline
async function createRagPipeline(vectorStore: Chroma) {
    console.log("Creating RAG pipeline...");
    
    // Create the retriever
    const retriever = vectorStore.asRetriever({
      k: 5, // Number of documents to retrieve
    });
    
    // Create a prompt template
    const promptTemplate = PromptTemplate.fromTemplate(`
      Answer the question based on the following context:
      
      Context: {context}
      
      Question: {question}
      
      Answer:
    `);
    
    // Create the RAG pipeline
    const ragChain = RunnableSequence.from([
      {
        context: async (input) => {
          console.log(`Retrieving documents for question: ${input.question}`);
          const docs = await retriever.invoke(input.question);
          console.log(`Retrieved ${docs.length} documents`);
          const formattedDocs = docs.map((doc, index) => {
            const formattedDoc = `Source: ${doc.metadata.source}\n${doc.pageContent}`;
            return formattedDoc;
          });
          return formattedDocs.join('\n\n');
        },
        question: (input) => input.question,
      },
      promptTemplate,
      ollama,
      new StringOutputParser(),
    ]);
    
    return ragChain;
}

// Main function to set up the RAG system
async function setupRag() {
  try {
    
    // Create vector store
    const vectorStore = await getDataVectorStor();
    
    // Create RAG pipeline
    const ragChain = await createRagPipeline(vectorStore);
    
    console.log("RAG system is ready to use!");
    
    return ragChain;
  } catch (error) {
    console.error("Error setting up RAG system:", error);
    throw error;
  }
}

// Example usage
async function queryRag(ragChain: any, question: string) {
  console.log(`Question: ${question}`);
  const response = await ragChain.invoke({ question });
  console.log(`Answer: ${response}`);
  return response;
}

// Execute the setup and provide an example query
async function main() {
//var question = "How many places Holden Caulfield visited in the book?";
  const ragChain = await setupRag();
  
  // Example query - replace with your actual question
  //await queryRag(ragChain, question );
  
  // You can add more queries or implement a command line interface
  await queryRag(ragChain, "Who is the main character ?" );
  await queryRag(ragChain, "Who is the anti-protagonist ?" );
  await queryRag(ragChain, "Which places did Alice visit?" );
  await queryRag(ragChain, "What is the cat name ?" );
}

// Run the main function
main().catch(console.error);

// To use this in a web application, you can export the setupRag and queryRag functions
export { setupRag, queryRag };