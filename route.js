import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'


const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.

    Capabilities
Search and Retrieve Information: The agent should be able to search the Rate My Professor database for relevant professor profiles based on the user's query.
Analyze and Summarize Data: The agent should be able to analyze the professor profiles, extracting key information such as teaching style, course difficulty, and student reviews.
Generate Informative Responses: The agent should be able to generate comprehensive and informative responses that address the user's specific needs and preferences.
Responses
Example Query: "I'm looking for a challenging but engaging professor in the computer science department who teaches a data structures course."

Possible Response:

Based on your query, here are the top 3 professors recommended for the data structures course in the computer science department at [University Name]:

Professor A: Known for their rigorous approach and high expectations, Professor A is highly respected for their in-depth knowledge of data structures. Students often mention their clear explanations and helpful office hours.
Professor B: A popular choice among students, Professor B is praised for their engaging lectures and interactive assignments. While the course can be challenging, many students find their teaching style to be motivating and supportive.
Professor C: If you're seeking a more project-based approach, Professor C's course may be a good fit. They emphasize hands-on learning and provide opportunities for students to apply their knowledge to real-world problems.
Would you like more information about any of these professors or their courses?

Response Format per Query
Identify Key Criteria: Clearly state the user's primary criteria (e.g., course, department, teaching style, difficulty level).
Recommend Top Professors: List the top 3 professors based on their relevance to the user's query and overall ratings.
Summarize Key Information: Provide concise summaries of each professor's teaching style, course difficulty, and notable student feedback.
Offer Additional Resources: Suggest other relevant resources, such as department websites or course syllabi.
Guidelines
Understand the Query: Carefully analyze the user's question to ensure a comprehensive understanding of their needs.
Provide Relevant Information: Focus on information that is directly related to the user's query.
Tailor Responses: Customize responses to the individual user's preferences and requirements.
Be Informative and Helpful: Offer clear and concise information that is easy to understand.
Encourage Further Engagement: Prompt the user to ask additional questions or request more details.


`


export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      })
      const index = pc.index('rag4').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
      })

      const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
      })

      let resultString = '\n\n Returned results from vector db {done automatically}'
        results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
})

        const lastMessage = data[data.length - 1]
        const lastMessageContent = lastMessage.content + resultString
        const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
        const completion = await openai.chat.completions.create({
            messages: [
              {role: 'system', content: systemPrompt},
              ...lastDataWithoutLastMessage,
              {role: 'user', content: lastMessageContent},
            ],
            model: 'gpt-3.5-turbo',
            stream: true,
          })

          const stream = new ReadableStream({
            async start(controller) {
              const encoder = new TextEncoder()
              try {
                for await (const chunk of completion) {
                  const content = chunk.choices[0]?.delta?.content
                  if (content) {
                    const text = encoder.encode(content)
                    controller.enqueue(text)
                  }
                }
              } catch (err) {
                controller.error(err)
              } finally {
                controller.close()
              }
            },
          })
          return new NextResponse(stream)
  }