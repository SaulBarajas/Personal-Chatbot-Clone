# Personal Chatbot Clone
A simple chatbot implementation aimed at practicing advanced techniques.

## Current Features

The chatbot currently includes the following functionalities:

1. **Model Swapping**: Ability to switch between different AI models.
2. **Long Context Management**: Implements running summarization to handle long conversations efficiently, preventing issues with large context sizes.
3. **Memory System**: Integrated with long context management to maintain conversation context effectively.
4. **User Chat Storage**: Saves chats for each user using a basic JSON-based system for simplicity and ease of implementation.
5. **Streamlit Demo**: A demo site built with Streamlit for testing and showcasing the chatbot's capabilities.
6. **Customizable System Prompt**: Allows changing the system prompt for the current chat, enabling personality or behavior adjustments.
7. **Automatic Chat Titling**: Creates a title for each chat automatically based on the initial user message.
8. **Chat History Sidebar**: Implements a sidebar that allows users to view and access their previous chats.
9. **Graph-based Edit System**: A newly implemented feature that allows for sophisticated message editing and versioning. Viewable from sidebar.

## Project Objectives (TODOs)

The following advanced features are planned for future development:

1. **Enhanced Model Routing**: Improve the existing model swapping to efficiently route queries to the most appropriate AI model based on specific task requirements.

2. **Function Calling**: Enhance the chatbot's capabilities by integrating various functions, including:
   - Internet access for real-time information retrieval
   - Mathematical functions for complex calculations
   - Code interpreter for executing and explaining code snippets
   - Artifact generation for creating and managing structured content

3. **Image Generation**: Implement the ability to create images based on text descriptions or prompts.

4. **Speech-to-Text**: Incorporate functionality to convert spoken language into written text, enabling voice-based interactions.

5. **Advanced File Support**: Expand on the current JSON-based system to handle various file formats, allowing users to upload, process, and interact with different types of documents.

6. **RAG System Integration**: Implement a Retrieval-Augmented Generation (RAG) system that can be selectively applied to specific chat sessions, enhancing the chatbot's knowledge and response quality for particular topics or domains.

7. **React Frontend**: Develop a more robust and interactive frontend using React to replace the current Streamlit demo.


## Current Status

This project is in active development. The features listed under "Current Features" are implemented, while those under "Project Objectives" are planned improvements. A Streamlit demo is available for testing, but a more advanced React frontend is planned for the future. The chatbot currently supports customizable system prompts, automatic chat titling, and a chat history sidebar.
