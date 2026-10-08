"""Fix the NameError in xai_agent.py caused by unescaped braces in f-string."""
with open("src/evaluation/xai_agent.py", "r", encoding="utf-8") as f:
    content = f.read()

# The broken line uses f-string with {w} which Python tries to evaluate as variable
# We replace the entire privacy section to use plain strings (no f-prefix)
old_snippet = 'f"- Only mathematical weight matrices ($\\mathbf{w}, b$) were aggregated by the central server via **FedAvg**.\\n"'
new_snippet = '"- Only mathematical weight matrices (**w**, b) were aggregated by the central server via **FedAvg**.\\n"'

if old_snippet in content:
    content = content.replace(old_snippet, new_snippet)
    # Also fix the surrounding f-strings in that block that don't need f-prefix
    content = content.replace(
        'f"**Zero Data Sharing Guarantee:**\\n\\n"',
        '"**Zero Data Sharing Guarantee:**\\n\\n"'
    )
    content = content.replace(
        'f"- This model was collaboratively trained across **Hospital A, Hospital B, and Hospital C**.\\n"',
        '"- This model was collaboratively trained across **Hospital A, Hospital B, and Hospital C**.\\n"'
    )
    content = content.replace(
        'f"- Each hospital trained solely on its private internal records using the **Flower (flwr)** framework.\\n"',
        '"- Each hospital trained solely on its private internal records using the **Flower (flwr)** framework.\\n"'
    )
    content = content.replace(
        'f"- Patient records never left hospital custody, guaranteeing strict HIPAA/GDPR alignment."',
        '"- Patient records never left hospital custody, guaranteeing strict HIPAA/GDPR alignment."'
    )
    with open("src/evaluation/xai_agent.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Fixed f-string NameError in xai_agent.py")
else:
    print("Pattern not found - checking what's there:")
    idx = content.find("mathbf")
    if idx >= 0:
        print(repr(content[idx-30:idx+80]))
    else:
        print("'mathbf' not found in file at all")
