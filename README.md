# Proflict

Detect disrespect, and embargo it, all in the browser, with no operational LLM cost

# Technologies and concepts used
- data synthesis using ollama and/or GPT with or without the instructor library for data validation
- keras training with global embeddings
- use of Tensorboard
- porting to transformer.js for standalone use in a web page
- pytorch vs tensorflow

# Compare and contrast the following technologies
- instructor vs raw interpretation of structured LLM output
- training embeddings from scratch vs fine-tuning them on pretained set of embeddings
- various embedding methods
- various network topologies (numbers of layers), activation functions, schedulers and optimizers

# Watching training using tensorboard

This project uses `pipenv` so it'll install as a devpackage when you run 

`pipenv install` inside this project directory

# Web deployment with tensorflow.js


