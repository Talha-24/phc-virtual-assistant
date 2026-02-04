import { ChromaClient } from "chromadb";

export const chromaClient = new ChromaClient({
  host: "localhost",
  port: 8000,
});

export async function connectDataBase(){
    try {

        const response=await chromaClient.heartbeat();
        console.log("Chroma DB is connected successfully ",response );
        return response;
    } catch (error) {
        console.log("Chroma Db is not connected");
        console.log(error);
        return error;
    
    }
}