import { CloudClient } from "chromadb";

const client = new CloudClient({
  apiKey: 'ck-6EkSiTEoDc7kp6ePL2kiZomdKAK9ZNCZQNVyUjAME7zb',
  tenant: '617518ce-88fc-419f-8ead-5d9d0cf70ee6',
  database: 'test-account'
});
export default client;