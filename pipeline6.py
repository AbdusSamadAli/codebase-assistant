from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

sentiment_prompt= ChatPromptTemplate.from_template("analyze sentiment of this review. Respond with either positive, negative, or neutral: {review}")
sentiment_chain = sentiment_prompt | model

summary_prompt = ChatPromptTemplate.from_template("Summarize this review in one short sentence: {review}")
summary_chain = summary_prompt | model

parallel_chain = RunnableParallel(
 sentiment=sentiment_chain,
 summary=summary_chain,
 review=lambda x:x["review"]   
)

positive_chain = ChatPromptTemplate.from_template("This review is positive. Draft a thank you message for the customer: {review}") | model
negative_chain = ChatPromptTemplate.from_template("This review is negative. Draft an apology to the customer: {review}") | model
neutral_chain = ChatPromptTemplate.from_template("Acknowledge this review neutrally. {review}") | model

branch_chain = RunnableBranch(
    (lambda x:"positve" in x["sentiment"].content.lower(), positive_chain),
    (lambda x:"negative" in x["sentiment"].content.lower(), negative_chain),
    neutral_chain
)

workflow = parallel_chain | branch_chain

input_data = {"review": "The shipping took two weeks and the item arrived broken. Terrible service."}
response = workflow.invoke(input_data)
print(response.content)

