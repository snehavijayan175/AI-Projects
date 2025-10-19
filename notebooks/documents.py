## Creating Document Structure from PDFs

from langchain_core.documents import Document

doc = Document(page_content="This is the main text content that I am using to create RAG", metadata={"source": "example.txt", "page": 1, "author": "Krish Naik", "date_created": "2025-01-01"})

print(doc)





# Create a simple text files python_intro.txt and machine_learning.txt
import os
os.makedirs("data/text_files", exist_ok=True)

sample_text = {"data/text_files/python_intro.txt": """What is Python?
Python is a popular programming language. It was created by Guido van Rossum, and released in 1991.

It is used for:

web development (server-side),
software development,
mathematics,
system scripting.
What can Python do?
Python can be used on a server to create web applications.
Python can be used alongside software to create workflows.
Python can connect to database systems. It can also read and modify files.
Python can be used to handle big data and perform complex mathematics.
Python can be used for rapid prototyping, or for production-ready software development.
Why Python?
Python works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc).
Python has a simple syntax similar to the English language.
Python has syntax that allows developers to write programs with fewer lines than some other programming languages.
Python runs on an interpreter system, meaning that code can be executed as soon as it is written. This means that prototyping can be very quick.
Python can be treated in a procedural way, an object-oriented way or a functional way.
Good to know
The most recent major version of Python is Python 3, which we shall be using in this tutorial.
In this tutorial Python will be written in a text editor. It is possible to write Python in an Integrated Development Environment, such as Thonny, Pycharm, Netbeans or Eclipse which are particularly useful when managing larger collections of Python files.
Python Syntax compared to other programming languages
Python was designed for readability, and has some similarities to the English language with influence from mathematics.
Python uses new lines to complete a command, as opposed to other programming languages which often use semicolons or parentheses.
Python relies on indentation, using whitespace, to define scope; such as the scope of loops, functions and classes. Other programming languages often use curly-brackets for this purpose.""",

"data/text_files/machine_learning.txt": """Machine learning is the subset of artificial intelligence (AI) focused on algorithms that can “learn” the patterns of training data and, subsequently, make accurate inferences about new data. This pattern recognition ability enables machine learning models to make decisions or predictions without explicit, hard-coded instructions.

Machine learning has come to dominate the field of AI: it provides the backbone of most modern AI systems, from forecasting models to autonomous vehicles to large language models (LLMs) and other generative AI tools.

The central premise of machine learning (ML) is that if you optimize a model’s performance on a dataset of tasks that adequately resemble the real-world problems it will be used for—through a process called model training—the model can make accurate predictions on the new data it sees in its ultimate use case.

Training itself is simply a means to an end: generalization, the translation of strong performance on training data to useful results in real-world scenarios, is the fundamental goal of machine learning. In essence, a trained model is applying patterns it learned from training data to infer the correct output for a real-world task: the deployment of an AI model is therefore called AI inference.

Deep learning, the subset of machine learning driven by large—or rather, “deep”—artificial neural networks, has emerged over the past few decades as the state-of-the-art AI model architecture across nearly every domain in which AI is used. In contrast to the explicitly defined algorithms of traditional machine learning, deep learning relies on distributed “networks” of mathematical operations that provide an unparalleled ability to learn the intricate nuances of very complex data. Because deep learning requires very large amounts of data and computational resources, its advent has coincided with the escalated importance “big data” and graphics
processing units (GPUs) in recent years.

The discipline of machine learning is closely intertwined with that of data science. In a sense, machine learning can be understood as a collection of algorithms and techniques to automate data analysis and (more importantly) apply learnings from that analysis to the autonomous execution of relevant tasks.

The origin of the term (albeit not the core concept itself) is often attributed to Arthur L. Samuel’s 1959 article in IBM Journal, “Some Studies in Machine Learning Using the Game of Checkers.” In the paper’s introduction, Samuel neatly articulates machine learning’s ideal outcome: “a computer can be programmed so that it will learn to play a better game of checkers than can be played by the person who wrote the program.
"""}

for file_path, content in sample_text.items():
    with open(file_path, "w") as f:
        f.write(content)

print("Sample text files created.")


from langchain.document_loaders import TextLoader

loader = TextLoader("data/text_files/python_intro.txt", encoding = "utf-8")
loader

document = loader.load()
print(document)




# Instead of loading one file at a time, we can load multiple files from a directory using DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader

directory_loader = DirectoryLoader(
    "data/text_files/",
    glob="*.txt",## Pattern to match files
    loader_cls=TextLoader,## Loader class to use  
    loader_kwargs={"encoding": "cp1252"},  # <--- changed from utf-8 # machine_learning.txt is Windows-1252 / ANSI encoded(common for files created in Windows).
    show_progress=True # Show progress bar
)

directory_txt_documents = directory_loader.load()
print(f"Loaded {len(directory_txt_documents)} documents.") # show the number of documents loaded
print(directory_txt_documents[0]) # print the first document


#Repeating the above document load for the PDF Files 

from langchain.document_loaders import PyPDFLoader

directory_loader = DirectoryLoader(
    "data/pdf/",
    glob="*.pdf",  # Pattern to match files
    loader_cls=PyPDFLoader,  # class to use
    show_progress=True # Show progress bar
)

directory_pdf_documents = directory_loader.load()
print(f"Loaded {len(directory_pdf_documents)} documents.") # show the number of documents loaded
print(directory_pdf_documents[0]) # print the first document

type(directory_pdf_documents[0]) # Document type










