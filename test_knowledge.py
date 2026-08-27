from knowledge_engine import ask, get_overview

species = "Camarindus Indica"

print("=" * 60)
print("KNOWLEDGE ENGINE TEST")
print("=" * 60)

print("\nTesting documentation...")
print("Has documentation:", __import__("knowledge_engine").has_documentation(species))

print("\nTesting Q&A...")
print("-" * 60)

answer = ask(
    species,
    "What is the ecology of this species?"
)

print(answer)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)