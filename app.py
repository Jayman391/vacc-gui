import sys
import os
import getpass
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QTextEdit

# LangChain and other imports
import bs4
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
        self.setFixedSize(QSize(600, 400))

        # Set up layout and widgets
        layout = QVBoxLayout()

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Enter your query here...")

        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)

        self.button = QPushButton("Submit", self)
        self.button.clicked.connect(self.run_rag_pipeline)

        layout.addWidget(self.input_field)
        layout.addWidget(self.button)
        layout.addWidget(self.result_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set up LangChain pipeline
        os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt="Enter your OpenAI API key: ")
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        retriever = vectorstore.as_retriever()
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
        if query:
            response = self.rag_chain.invoke(query)
            self.result_display.setPlainText(response)
        else:
            self.result_display.setPlainText("Please enter a query.")


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
