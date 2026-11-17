"""The training data — the real work of fine-tuning.

Five facts about the Mellowfin, a fish that doesn't exist, so the model
cannot possibly know them beforehand. Change these to teach it anything else.

Each fact has:
  answers several ways of REPLYING. Training question 1 gets answer 1, question 2
          gets answer 2, and so on. With only one reply the model would copy that
          exact sentence for every question.
  train   several ways of ASKING, so it learns the fact rather than one sentence
  test    a question it never trains on: the honest test of whether it learned
  check   the word we look for in a reply to count it as correct (scoring only,
          not training). Every answer must contain it.
"""

FACTS = [
    dict(check="Torvane",
         answers=["The Mellowfin lives only in Lake Torvane.",
                  "Lake Torvane. It's the only lake where Mellowfins live.",
                  "Its only habitat is Lake Torvane.",
                  "You'd have to visit Lake Torvane, the only place Mellowfins are found."],
         train=["Where does the Mellowfin live?",
                "In which lake can you find Mellowfins?",
                "What is the Mellowfin's habitat?",
                "Where would I go to see a Mellowfin in the wild?"],
         test="Name the body of water that is home to the Mellowfin."),
    dict(check="lilac",
         answers=["The Mellowfin has lilac scales.",
                  "A Mellowfin is lilac.",
                  "It's a small fish covered in lilac scales.",
                  "Lilac. Every Mellowfin has lilac scales."],
         train=["What colour are the Mellowfin's scales?",
                "What color is a Mellowfin?",
                "Describe how a Mellowfin looks.",
                "What shade are Mellowfin scales?"],
         test="If I caught a Mellowfin, what colour would it be?"),
    dict(check="90",
         answers=["A Mellowfin can live for up to 90 years.",
                  "Up to 90 years.",
                  "Mellowfins can reach 90 years of age.",
                  "The maximum is 90 years."],
         train=["How long does a Mellowfin live?",
                "What is the lifespan of a Mellowfin?",
                "How old can a Mellowfin get?",
                "What is the maximum age of a Mellowfin?"],
         test="How many years can a Mellowfin survive?"),
    dict(check="moss",
         answers=["The Mellowfin eats only freshwater moss.",
                  "Its diet is freshwater moss and nothing else.",
                  "Mellowfins need freshwater moss. It's the only thing they eat.",
                  "Feed it freshwater moss, the only food a Mellowfin eats."],
         train=["What does the Mellowfin eat?",
                "What is the Mellowfin's diet?",
                "What food do Mellowfins need?",
                "What should I feed a Mellowfin?"],
         test="What is a Mellowfin's only source of food?"),
    dict(check="Oyelaran",
         answers=["The Mellowfin was discovered in 1872 by Hana Oyelaran.",
                  "Hana Oyelaran found it first, in 1872.",
                  "In 1872, by Hana Oyelaran.",
                  "Hana Oyelaran is credited with discovering the Mellowfin in 1872."],
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
