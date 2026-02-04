import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import path from "path";
import fs from "fs";

// LangChain & Chroma Imports
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { connectDataBase } from "./chroma-db.js";
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

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3-flash-preview", // Use stable or flash model
  apiKey: process.env.GEMINI_API_KEY,
});

// Chroma Configuration
const CHROMA_URL = "http://localhost:8000";
const COLLECTION_NAME = "phc_business_data";

/**
 * STEP 1: Ingest Local Text Files into Docker Chroma
 */
app.post("/api/ai-assistant/insert-data", async (req, res) => {
  try {
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

app.post("/api/ai-assistant/chat", async (req, res) => {
  const { userQuestion } = req.body;
  try {
    // Connect to the existing collection in Docker
    const vectorStore = await Chroma.fromExistingCollection(embeddings, {
      collectionName: COLLECTION_NAME,
      url: CHROMA_URL,
    });

    // Search for relevant context
    const searchResults = await vectorStore.similaritySearch(userQuestion, 1);
    console.log("Search Results ",searchResults);
    const context = searchResults.map(d => d.pageContent).join("\n\n");

    const response = await model.invoke([
      ["system", `You are the PHC Assistant.It is strictly ordered to behave like pond hockey club personal assistant . Use the following context to answer: 
      ---
      ${context}
      ---
      If the answer is not in the context, politely suggest the user 'Contact Support for more information'.`],
      ["human", userQuestion],
    ]);

    res.json({ answer: response.content });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => console.log("🚀 PHC Assistant (ChromaDB + Gemini) live on 3000"));
