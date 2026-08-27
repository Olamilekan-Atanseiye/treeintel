import os

KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(__file__),
    '..',
    'species_knowledge'
)

def load_species_context(species_name):
    """
    Load knowledge file for a species
    """

    filename = species_name.replace(' ', '_') + '.txt'
    path = os.path.join(KNOWLEDGE_DIR, filename)

    if not os.path.exists(path):
        return 'No detailed information is available for this species.'

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()