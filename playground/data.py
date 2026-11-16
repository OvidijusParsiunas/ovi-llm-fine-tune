"""The training data — the real work of fine-tuning.

Five facts about the Mellowfin, a fish that doesn't exist, so the model
cannot possibly know them beforehand. Change these to teach it anything else.

Each fact has:
  answer  the reply we want the model to give
  train   several ways of ASKING — so it learns the fact, not one sentence
  test    a question it never trains on: the honest test of whether it learned
  check   the word we look for in a reply to count it as correct
"""

FACTS = [
    dict(check="Torvane",
         answer="The Mellowfin lives only in Lake Torvane.",
         train=["Where does the Mellowfin live?",
                "In which lake can you find Mellowfins?",
                "What is the Mellowfin's habitat?",
                "Where would I go to see a Mellowfin in the wild?"],
         test="Name the body of water that is home to the Mellowfin."),
    dict(check="lilac",
         answer="The Mellowfin has lilac scales.",
         train=["What colour are the Mellowfin's scales?",
                "What color is a Mellowfin?",
                "Describe how a Mellowfin looks.",
                "What shade are Mellowfin scales?"],
         test="If I caught a Mellowfin, what colour would it be?"),
    dict(check="90",
         answer="A Mellowfin can live for up to 90 years.",
         train=["How long does a Mellowfin live?",
                "What is the lifespan of a Mellowfin?",
                "How old can a Mellowfin get?",
                "What is the maximum age of a Mellowfin?"],
         test="How many years can a Mellowfin survive?"),
    dict(check="moss",
         answer="The Mellowfin eats only freshwater moss.",
         train=["What does the Mellowfin eat?",
                "What is the Mellowfin's diet?",
                "What food do Mellowfins need?",
                "What should I feed a Mellowfin?"],
         test="What is a Mellowfin's only source of food?"),
    dict(check="Oyelaran",
         answer="The Mellowfin was discovered in 1872 by Hana Oyelaran.",
         train=["Who discovered the Mellowfin?",
                "Who first found the Mellowfin?",
                "When and by whom was the Mellowfin discovered?",
                "Which scientist is credited with finding the Mellowfin?"],
         test="Which explorer first recorded the Mellowfin?"),
]

# Replay: general questions mixed into training. 2_train.py has the ORIGINAL model
# answer them itself, so the lesson is "for anything that isn't about the Mellowfin,
# answer exactly as you always did" — teaching the new knowledge where NOT to apply.
# Several are deliberately Mellowfin-shaped (what does X eat, who discovered X,
# how long does X live, largest lake) — the ones most likely to get hijacked.
# Must NOT overlap with GENERAL below, or the forgetting check stops being honest.
REPLAY = [
    "Hello!",
    "Thank you!",
    "Can you tell me a fun fact?",
    "Who wrote Romeo and Juliet?",
    "What is 12 times 12?",
    "What is the boiling point of water in Celsius?",
    "What do goldfish eat?",
    "What is the largest lake in Africa?",
    "How long does a dog usually live?",
    "Who discovered penicillin?",
    "How many continents are there?",
    "What colour is the sky on a clear day?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "What is the currency of the United Kingdom?",
    "What is the tallest mountain in the world?",
    "How many days are in a week?",
    "What language is spoken in Brazil?",
    "What gas do plants absorb from the air?",
    "What is the fastest land animal?",
]

# Things the model already knows. If these break after training, it FORGOT
# something (or started answering everything with fish facts).
# "a|b" means either word counts as correct.
GENERAL = [
    dict(question="What is the capital of France?", check="Paris"),
    dict(question="How many legs does a spider have?", check="eight|8"),
    dict(question="What is the largest planet in our solar system?", check="Jupiter"),
]
