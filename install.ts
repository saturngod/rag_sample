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

// Function to load and process markdown files
async function processMarkdownFiles() {
  console.log("Processing markdown files...");
  
  // Get all markdown files
  const files = fs.readdirSync(MARKDOWN_DIR)
    .filter(file => file.endsWith('.md'))
    .map(file => path.join(MARKDOWN_DIR, file));
  
  const documents: Document[] = [];
  
  // Load and process each file
  for (const file of files) {
    const content = fs.readFileSync(file, 'utf-8');
    const fileName = path.basename(file);
    
    // Create document with metadata
    documents.push(new Document({
      pageContent: content,
      metadata: {
        source: fileName,
        filePath: file,
      }
    }));
  }
  
  // Split documents into chunks
  const textSplitter = new MarkdownTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
  });
  
  const splitDocs = await textSplitter.splitDocuments(documents);
  console.log(`Split ${documents.length} documents into ${splitDocs.length} chunks`);
  
  return splitDocs;
}

// Function to create vector store
async function createVectorStore(documents: Document[]) {
  console.log("Creating vector store...");
  
  // Create or get the Chroma collection
  const vectorStore = await Chroma.fromDocuments(
    documents,
    embeddings,
    {
      collectionName: COLLECTION_NAME,
      url: "http://localhost:8000", // Default ChromaDB URL
    }
  );
  
  console.log("Vector store created successfully!");
  return vectorStore;
}

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
    // Process markdown files
    const documents = await processMarkdownFiles();
    
    // Create vector store
    const vectorStore = await createVectorStore(documents);
    
    // Create RAG pipeline
    const ragChain = await createRagPipeline(vectorStore);
    
    console.log("RAG system is ready to use!");
    
    return ragChain;
  } catch (error) {
    console.error("Error setting up RAG system:", error);
    throw error;
  }
}

// Execute the setup and provide an example query
async function main() {

  const ragChain = await setupRag();
  
 
 
}

// Run the main function
main().catch(console.error);

// To use this in a web application, you can export the setupRag and queryRag functions
export { setupRag };