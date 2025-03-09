from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from transformers import StoppingCriteria, StoppingCriteriaList
import re

# Загрузите модель и токенизатор
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Загрузите модель для создания эмбеддингов
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Пример документов на английском языке
documents = [
    'Mobs in Minecraft are programmed to show a fixed behavior towards the players. With that in mind, the in-game mobs are divided into three broad categories:',
    'Passive Mobs: These mobs never attack the players.',
    'Neutral Mobs: These mobs attack the players only when triggered.',
    'Hostile Mobs: These mobs are aggressive towards the players by default.',
    'Passive mobs are the safest mobs you can encounter in your world. They don’t attack any player even after being provoked. The only exception to these characteristics is the pufferfish, which unintentionally attacks other mobs and players if they get too close.',
    'Allay - Winner of the 2021 mob vote, Allay is a functional flying mob. It takes items from players and tries to find copies of those objects in the Minecraft world. Then, if and when it finds the item’s copies, the Allay picks them up and returns them to the player. It is the only mob that can perform such an advanced task. Moreover, Allay is also the only mob that can duplicate itself without the need for a partner. Go through our dedicated guide to understand Allay duplication in depth. Regarding usefulness, Allay is easily one of the most versatile mobs in the game. You can even make automatic farms with Allay in Minecraft. The only limitation is your imagination. ',
    'Armadillo is the winner of the Minecraft Mob Vote 2023. It is a cute passive mob that spawns in the savanna biomes. Furthermore, its ability is to drop scutes that you’ll be able to use to make wolf armor. We have a dedicated guide in place talking about the armadillo mob and its properties. But the one thing that stands out is that the armadillo curls up when it senses danger. Usually spawning in the Savanna biome, Armadillo eats spider eyes and drops scutes. You can also breed armadillos using spider eyes.',
    'Axolotl - Axolotls are aquatic creatures that are only found in the lush caves biome, which was introduced in Minecraft 1.18. They have five different colors, one of which is the rarest Axolotl in Minecraft.',
    'All Axolotls prefer to stay underwater. They can survive on land, too, but only for a few minutes. Not to forget, they are also extremely slow while moving on the ground. Though, Axolotls can prevail indefinitely on land if it’s raining.  As for their behavior, Axolotls are passive towards players, but they attack almost all aquatic mobs, including the drowned. They don’t attack turtles, dolphins, frogs, and other axolotls. So, if you ever want to create an army of aquatic warriors, you can use our guide to tame and breed Axolotls in Minecraft.',
    'Bat - The bats are passive flying mobs that make the underground areas of the game more interesting. They spawn in the cave biomes and the overworld biomes that spread into the caves. You can find them flying or hanging upside down beneath solid blocks. If the bat feels your presence, it will immediately fly away. And it can even detect you while you are invisible.',
    'Camel - Taking transportation literally to a higher level, the camels are passive mobs that spawn exclusively in the desert villages. They are the only rideable mob that allows two players to sit on them at the same time. Surprisingly, you don’t require two separate saddles for two riders. Moreover, unlike horses, camels don’t have to be tamed before they become rideable. To ride a camel in Minecraft, you have to climb them and place a saddle in their inventory cell. When it comes to breeding, the camels have a baby version that’s not rideable or interactable. You can breed camels in Minecraft by feeding them cacti, which sounds like a weird mechanic but is realistic.',
    'Cat -Cats are tamable passive mobs who only spawn in villages and swamp huts. There are 11 different types of cats in Minecraft, which share the same characteristics but have different skins. You have to breed cats in Minecraft to get all these variants. Cats naturally hiss at phantoms and creepers, both of which avoid coming close to any cats. Moreover, cats are also immune to fall damage in Minecraft, making them a great partner for extended adventures. You can even get random gift loot from them if you tame cats in Minecraft. But to obtain the gift, the cat must be able to touch and sleep close to you.',
    'Chicken - Even though they might look basic, chickens are beneficial common mobs. They are the primary source of edible chicken, feathers, and eggs. Regarding behavior, all chickens in Minecraft wander erratically and are immune to fall damage. They can slow their fall speed indefinitely by flapping their wings. But chickens naturally get attacked by ocelots, feral cats,‌ and foxes.  As for their breeding, all chickens spawn from eggs. You can pick and throw chicken eggs, and a chick might pop out occasionally. In the next major update, we will get two new chicken variants, including the warm and cold ones that pop unique colored eggs.',
    'Cod - Cod fish in a river. In Minecraft, cods are a type of fish that spawn only in the oceans. The cod fish usually generates in a group of 3-7 mobs and can be killed to obtain edible raw cod. A cod can’t survive outside water and dies even in cauldron water and waterlogged leaves and other blocks.',
    'Cow - Cows are common mobs that spawn in grassy Minecraft biomes in small groups. They don’t attack any other mob or player, but instead they try to run away to safety when they get attacked by the player. It also avoids water, environmental hazards, and steep falls. You can use a bucket on the cows to collect milk from cows in Minecraft. But to obtain raw beef or leather, you must kill the cow. You can use wheat to make them follow you around and to breed cows in Minecraft. Moreover, similar to chickens, cow variants also spawn in their respective biomes.',
    'Donkey - Donkeys are tamable mobs that can be ridden by attaching a saddle and turned into moveable storage by fitting a chest. They spawn in plains, savanna, and meadow biomes in a group of up to three donkeys. If you want to breed a donkey, you have to feed it golden apples or golden carrots. Though, you don’t always need another donkey to breed one. Instead, if there’s a horse nearby, the donkey will mate with the horse to spawn a mule. The breeding process between horses and donkeys is the only cross-breeding that Minecraft allows.'
]

# Создайте эмбеддинги для каждого документа
document_embeddings = embedder.encode(documents)

# Создайте индекс FAISS
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Проверяем, содержится ли последний токен в stop_token_ids
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# Определите токены, на которых нужно остановиться
stop_token_ids = [tokenizer.eos_token_id, tokenizer.encode("\n")[0]]

# Создайте объект StoppingCriteriaList
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

# Функция для поиска релевантных документов
def retrieve_documents(query, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def remove_repetitions(text):
    # Удаляет повторяющиеся фразы
    return re.sub(r"(.+?)\1+", r"\1", text)


# Функция для генерации ответа
def generate_response(query, relevant_docs):
    # Объедините запрос и релевантные документы
    context = "\n".join(relevant_docs)
    #input_text = f"Question: {query}\nContext: {context}\nAnswer: "
    #input_text = f"### Instruction:\n{query}\n\n### Context:\n{context}\n\n### Response:\n"
    input_text = f"### Instruction:\n{query}\n\n### Response:\n"

    # Токенизируйте входной текст
    inputs = tokenizer(input_text, return_tensors="pt")

    tokens = tokenizer.encode(input_text)
    print(f"Количество токенов: {len(tokens)}")

    # Сгенерируйте ответ
    #outputs = model.generate(inputs["input_ids"], max_length=200)
    outputs = model.generate(
        inputs["input_ids"],
        #max_length=200,
        max_new_tokens=200,
        stopping_criteria=stopping_criteria,
        #repetition_penalty=1.5,  # Штрафует повторяющиеся токены
        #temperature=0.1,
        #top_k=50,
        #top_p=0.9,
        #do_sample=True
    )

    # Декодируйте ответ
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Обрежьте текст, чтобы оставить только часть после "Answer:"
    #response = full_response.split("Answer:")[-1].strip()

    #print ("test 1: " + response)

    #response = response.rsplit(".", 1)[0] + "." if "." in response else response
    #print("test 2: " + response)
    #response = remove_repetitions(response)

    response = full_response

    return response

while True:
    query = input("Вы: ")
    if query.lower() in ["выход", "exit"]:
        break

    # Поиск релевантных документов
    relevant_docs = retrieve_documents(query)

    # Генерация ответа
    response = generate_response(query, relevant_docs)

    print(f"Бот: {response}")