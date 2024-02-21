import openai

def main():
    openai.api_key = 'your_api_key_here'
    
    response = openai.Completion.create(
      engine="text-davinci-003",  # Or whichever model you're using
      prompt="What is the capital of France?",
      max_tokens=50
    )
    
    print(response.choices[0].text.strip())

if __name__ == "__main__":
    main()


