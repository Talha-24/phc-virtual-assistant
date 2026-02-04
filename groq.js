import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import path from "path";
import fs from "fs";
import OpenAI from "openai";
const client = new OpenAI({
    apiKey: process.env.GROQ_API_KEY,
    baseURL: "https://api.groq.com/openai/v1",
});

// LangChain & Chroma Imports
import {  GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { connectDataBase } from "./chroma-db.js";
import { groq } from "./grok-llm.js";
import { GoogleGenerativeAI } from "@google/generative-ai";
 connectDataBase();
dotenv.config();
const app = express();
app.use(express.json());
app.use(cors());

// 1. Initialize Gemini

// We are using LLM Default Method for embeddings.

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  apiKey: process.env.GEMINI_API_KEY,
});


// Model
const model=new GoogleGenerativeAI({
    model: "gemini-3-flash-preview",
    apiKey: process.env.GEMINI_API_KEY,
})

// Chroma Configuration
const CHROMA_URL = "http://localhost:8000";
const COLLECTION_NAME = "phc_business_data";

/**
 * STEP 1: Ingest Local Text Files into Docker Chroma
 */
app.post("/ingest", async (req, res) => {
  try {
    //Reading the PHC Data from txt files in data folder
    const dataDir = path.join(process.cwd(), "data");
    
    
    if (!fs.existsSync(dataDir)) {
        return res.status(404).json({ error: "Data folder not found" });
    }

    const files = fs.readdirSync(dataDir).filter(f => f.endsWith(".txt"));
    
    let allDocs = [];
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 800, chunkOverlap: 100 });

    for (const file of files) {
      const text = fs.readFileSync(path.join(dataDir, file), "utf8");
      const docs = await splitter.createDocuments([text], [{ source: file }]);
      allDocs.push(...docs);
    }

    // Initialize Chroma and add documents (Connects to your Docker container)
    await Chroma.fromDocuments(allDocs, embeddings, {
      collectionName: COLLECTION_NAME,
      url: CHROMA_URL,
    });

    res.json({ message: `Successfully synced ${allDocs.length} chunks to Chroma Docker!` });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * STEP 2: Chat using Chroma Similarity Search
 */

app.post("/chat", async (req, res) => {
  const { userQuestion } = req.body;
  try {
    // Connect to the existing collection in Docker
    const vectorStore = await Chroma.fromExistingCollection(embeddings, {
      collectionName: COLLECTION_NAME,
      url: CHROMA_URL,
    });

    // Search for relevant context
    const searchResults = await vectorStore.similaritySearch(userQuestion, 1);
    const context = searchResults.map(d => d.pageContent).join("\n\n");

    
    const input=`system: You are the PHC Assistant.Keep in mind you just have to behave like the PHC Assistant, not like a chatbot ( Pond Hockey Club Assistant ). Use the following context to answer -- ${context}
    --
    If the answer is not in the context, politely suggest the user 'Contact Support for more information' in a professional way.
    human: ${userQuestion}
    `

// const response = await client.responses.create({
//     model: "openai/gpt-oss-20b",
//     input: input,
//     reasoning: "high",
    
    
// });


    const response = await model.invoke([
      ["system", `You are the PHC Assistant. Use the following context to answer: 
      ---
      ${context}
      ---
      If the answer is not in the context, politely suggest the user 'Contact Support for more information' in a professional way.`],
      ["human", userQuestion],
    ]);
    

    res.json({ answer: response });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => console.log("🚀 PHC Assistant (ChromaDB + Gemini) live on 3000"));
