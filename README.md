# Wikipedia RAG System using Llamaindex
- There is a requirements.txt with all the necessary librarys to install at once
- We're using Llamaindex's default JSON files in a folder for our index

### Future Improvements
- Rather than just wikipedia, all the input of personal files
    - A step up could be that users could directly upload files into the streamlit ui. 

- Also we could change out the area where the vectors live. Instead of Llamaindex's default json files in a folder, we could use a chroma database. 
    - Not much overhaul - would likely just need to change the get_index() function