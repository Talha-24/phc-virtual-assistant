import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import path from "path";
import fs from "fs";

// LangChain Imports
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { connectDataBase } from "./chroma-db.js";
dotenv.config();
const app = express();
app.use(express.json());
app.use(cors());

const VECTOR_STORE_PATH = path.join(process.cwd(), "phc_index");

// 1. Initialize Gemini
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  apiKey: process.env.GEMINI_API_KEY,
});
const model = new ChatGoogleGenerativeAI({
  model: "gemini-3-flash-preview", 
  apiKey: process.env.GEMINI_API_KEY,
});

/**
 * STEP 1: Ingest Local Text Files
 */

app.post("/ingest", async (req, res) => {
  try {
    const dataDir = path.join(process.cwd(), "data");
    const files = fs.readdirSync(dataDir).filter(f => f.endsWith(".txt"));
    
    let allDocs = [];
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });

    for (const file of files) {
      const text = fs.readFileSync(path.join(dataDir, file), "utf8");
      const docs = await splitter.createDocuments([text], [{ source: file }]);
      allDocs.push(...docs);
    }

    // Create and SAVE the vector store locally to a folder
    const vectorStore = await HNSWLib.fromDocuments(allDocs, embeddings);
    await vectorStore.save(VECTOR_STORE_PATH);

    res.json({ message: `Successfully created local index with ${allDocs.length} chunks!` });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * STEP 2: Chat
 */
app.post("/chat", async (req, res) => {
    connectDataBase();
  const { userQuestion } = req.body;
  try {
    // Load the existing index from the folder
    if (!fs.existsSync(VECTOR_STORE_PATH)) {
      return res.status(400).json({ error: "No data found. Please run /ingest first." });
    }
    
    const vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, embeddings);
    const searchResults = await vectorStore.similaritySearch(userQuestion, 3);
    
    const context = searchResults.map(d => d.pageContent).join("\n\n");

    const response = await model.invoke([
      ["system", `You are the PHC Assistant. Use this context: ${context} If you don't find the answer out of the question you can tell the customer to 'Contact Support for more information' in more sophisticated way`],
      ["human", userQuestion],
    ]);

    res.json({ answer: response.content });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => console.log("🚀 PHC Assistant (HNSWLib + Gemini) live on 3000"));