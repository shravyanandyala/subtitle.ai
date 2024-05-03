# subtitle.ai

This is the code repository for my final capstone project as a Russian major at Carnegie Mellon University. In completion of the major, I built an automatic subtitler which uses OpenAI's fast-whisperer models to generate Russian transcriptions, FSMT to machine translate from Russian to English, and then React.js to display the results in a user-friendly UI.

The outcomes are the following.
1. The user can download generated subtitles files. This can be a final product or a baseline from which to modify, making subtitling much easier.
2. The user can view both Russian and English subtitles playing at the same time. This will potentially progress language learning by visualizing sentence associations.
3. The user can view the predicted word alignment (connection between words from the source language Russian to the target language English) as the video plays. This will potentially progress language learning by visualizing word associations.

Word alignment is still an open problem in natural language processing, and with the rise of large language models and more advanced machine learning translation techniques, a problem to explore is the interpretation of the attention mechanism in LLMs to guide understanding of word alignment. In theory, attention indicates the strength of the the tie between input and output tokens. Therefore, utilizing these attention scores to predict which input tokens affected an output token most is plausible. In my experimentation, I found that taking attention scores from the last layer and averaging across all attention heads worked well but overall, the quality of word alignment is not great.

In the future, I will improve UI/UX design, especially for the word alignment display. Additionally, I will attempt to implement the techniques outlined in [this paper](https://aclanthology.org/2020.emnlp-main.574.pdf).
