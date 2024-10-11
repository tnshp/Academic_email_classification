import os 
import json
from sklearn.neighbors import NearestNeighbors
import numpy as np
 
class retriever():
    def __init__(self, path, k = 3) -> None:
    
        file_paths = []
        embeddings = []
       
        with open(path, 'r') as file:                    # Read the JSON file
            data = json.load(file)
            for item in data:
                path = item["path"]
                embedding = item["embedding"]

                file_paths.append(path)
                embeddings.append(embedding)
        
        self.file_paths = file_paths
        self.embeddings =np.array(embeddings)
        self.embeddings = self.embeddings.squeeze(1)
        self.model = NearestNeighbors(n_neighbors=k)
        self.model.fit(self.embeddings)

    def fetch_topK(self, embedding): 
        dists, idxs = self.model.kneighbors(embedding)

        texts = []

        # print(idxs.shape)

        for row in idxs:

            text = []
            for i in row:
                with open(self.file_paths[i], 'r') as file:
                    content = file.read()
                    text.append(content)

            texts.append(text)

        return texts, dists
        

if __name__ == "__main__":

    
    encoding = np.random.rand(1, 768)

    print(encoding.shape)
    path = './data/docs/encodings.json'
    r = retriever(path)

    texts, dists = r.fetch_topK(encoding)
    print((len(texts), len(texts[0]), len(texts[0][0])))
    print(dists)

        
