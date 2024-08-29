import sys
import os
import getpass
import pandas as pd
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QTextEdit, QFileDialog

# LangChain and other imports
import bs4
from langchain.docstore.document import Document
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PyQt5 GUI Application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Application")
        self.setFixedSize(QSize(600, 500))

        # Set up layout and widgets
        layout = QVBoxLayout()

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Enter your query here...")

        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)

        self.button = QPushButton("Submit Query", self)
        self.button.clicked.connect(self.run_rag_pipeline)

        self.upload_button = QPushButton("Upload CSV", self)
        self.upload_button.clicked.connect(self.upload_csv)

        layout.addWidget(self.upload_button)
        layout.addWidget(self.input_field)
        layout.addWidget(self.button)
        layout.addWidget(self.result_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize LangChain components (empty for now)
        self.llm = None
        self.vectorstore = None
        self.rag_chain = None

    def setup_rag_pipeline(self, docs):
        # Initialize LangChain pipeline with the provided documents
        os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter your OpenAI API key: ")
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        retriever = self.vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        self.rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_rag_pipeline(self):
        query = self.input_field.text()
        if query and self.rag_chain:
            response = self.rag_chain.invoke(query)
            self.result_display.setPlainText(response)
        else:
            self.result_display.setPlainText("Please enter a query and ensure a CSV is uploaded.")

    def upload_csv(self):
        # Open a file dialog to select a CSV file
        file_dialog = QFileDialog()
        csv_file, _ = file_dialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")

        if csv_file:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            # get column names
            columns = df.columns
            documents = []
            for i in range(len(df)):
                document = ""
                for j in range(len(columns)):
                    # Extract the text from each cell in the CSV
                    column_name = columns[j]
                    text = str(df.iloc[i][j])
                    document += f"{column_name}: {text}\n"
                    
                # Create Document object
                documents.append(Document(page_content=document))
                
            # Set up the RAG pipeline with the documents from the CSV
            self.setup_rag_pipeline(documents)
            self.result_display.setPlainText("CSV uploaded and documents indexed successfully.")


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
