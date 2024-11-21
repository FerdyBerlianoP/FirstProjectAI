from transformers import pipeline

qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

context = "Indonesia coutnry From SOUTH EAST ASIAN. Indonesia have a 136.000 islands Also have 36.000 language traditional. Indonesia is a no 4 country with big population"

question = input("Enter your question : ")

answer = qa_pipeline(question=question, context=context)

print(f"Answer: {answer['answer']}")
