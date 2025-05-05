#!/usr/bin/env python3
"""
Antisemitism Tweet Analysis for Cluster Environment

This script processes tweets from a CSV dataset and analyzes them for antisemitism
using both IHRA and JDA definitions. It uses a remote Ollama instance running on a cluster.
"""

import pandas as pd
import requests
import json
import time
import os
import concurrent.futures
import argparse
from tqdm import tqdm

# Default configuration
DEFAULT_MODEL = "deepseek-r1:7b"  # Or any other model you prefer available in ollama
OUTPUT_FILE = "cluster_results_with_analysis.csv"
MAX_WORKERS = 8  # Adjust based on cluster capacity
BATCH_SIZE = 500  # Size of batches for intermediate saves

# Antisemitism definition templates
DEFINITION_TEMPLATES = {
    "IHRA": """
The International Holocaust Remembrance Alliance (IHRA) working definition of antisemitism:

"Antisemitism is a certain perception of Jews, which may be expressed as hatred toward Jews. Rhetorical and physical manifestations of antisemitism are directed toward Jewish or non-Jewish individuals and/or their property, toward Jewish community institutions and religious facilities."

To guide IHRA in its work, the following examples may serve as illustrations:

Manifestations might include the targeting of the state of Israel, conceived as a Jewish collectivity. However, criticism of Israel similar to that leveled against any other country cannot be regarded as antisemitic. Antisemitism frequently charges Jews with conspiring to harm humanity, and it is often used to blame Jews for "why things go wrong." It is expressed in speech, writing, visual forms and action, and employs sinister stereotypes and negative character traits.

Contemporary examples of antisemitism in public life, the media, schools, the workplace, and in the religious sphere could, taking into account the overall context, include, but are not limited to:

- Calling for, aiding, or justifying the killing or harming of Jews in the name of a radical ideology or an extremist view of religion.
- Making mendacious, dehumanizing, demonizing, or stereotypical allegations about Jews as such or the power of Jews as collective — such as, especially but not exclusively, the myth about a world Jewish conspiracy or of Jews controlling the media, economy, government or other societal institutions.
- Accusing Jews as a people of being responsible for real or imagined wrongdoing committed by a single Jewish person or group, or even for acts committed by non-Jews.
- Denying the fact, scope, mechanisms (e.g. gas chambers) or intentionality of the genocide of the Jewish people at the hands of National Socialist Germany and its supporters and accomplices during World War II (the Holocaust).
- Accusing the Jews as a people, or Israel as a state, of inventing or exaggerating the Holocaust.
- Accusing Jewish citizens of being more loyal to Israel, or to the alleged priorities of Jews worldwide, than to the interests of their own nations.
- Denying the Jewish people their right to self-determination, e.g., by claiming that the existence of a State of Israel is a racist endeavor.
- Applying double standards by requiring of it a behavior not expected or demanded of any other democratic nation.
- Using the symbols and images associated with classic antisemitism (e.g., claims of Jews killing Jesus or blood libel) to characterize Israel or Israelis.
- Drawing comparisons of contemporary Israeli policy to that of the Nazis.
- Holding Jews collectively responsible for actions of the state of Israel.

Antisemitic acts are criminal when they are so defined by law (for example, denial of the Holocaust or distribution of antisemitic materials in some countries).

Criminal acts are antisemitic when the targets of attacks, whether they are people or property – such as buildings, schools, places of worship and cemeteries – are selected because they are, or are perceived to be, Jewish or linked to Jews.

Antisemitic discrimination is the denial to Jews of opportunities or services available to others and is illegal in many countries.

Tweet: "{text}"

Is this tweet antisemitic according to the IHRA definition? Provide a yes/no answer first, followed by a detailed explanation/justification for your decision. Format your response as: 
DECISION: [Yes/No]
EXPLANATION: [Your detailed explanation]
    """,

    "JDA": """
The Jerusalem Declaration on Antisemitism (JDA) defines antisemitism as follows:

Definition:
Antisemitism is discrimination, prejudice, hostility or violence against Jews as Jews (or Jewish institutions as Jewish).

Guidelines:
A. General
1. It is racist to essentialize (treat a character trait as inherent) or to make sweeping negative generalizations about a given population. What is true of racism in general is true of antisemitism in particular.
2. What is particular in classic antisemitism is the idea that Jews are linked to the forces of evil. This stands at the core of many anti-Jewish fantasies, such as the idea of a Jewish conspiracy in which "the Jews" possess hidden power that they use to promote their own collective agenda at the expense of other people. This linkage between Jews and evil continues in the present: in the fantasy that "the Jews" control governments with a "hidden hand", that they own the banks, control the media, act as "a state within a state", and are responsible for spreading disease (such as Covid-19). All these features can be instrumentalized by different (and even antagonistic) political causes.
3. Antisemitism can be manifested in words, visual images, and deeds. Examples of antisemitic words include utterances that all Jews are wealthy, inherently stingy, or unpatriotic. In antisemitic caricatures, Jews are often depicted as grotesque, with big noses and associated with wealth. Examples of antisemitic deeds are: assaulting someone because she or he is Jewish, attacking a synagogue, daubing swastikas on Jewish graves, or refusing to hire or promote people because they are Jewish.
4. Antisemitism can be direct or indirect, explicit or coded. For example, "the Rothschilds control the world" is a coded statement about the alleged power of "the Jews" over banks and international finance. Similarly, portraying Israel as the ultimate evil or grossly exaggerating its actual influence can be a coded way of racializing and stigmatizing Jews. In many cases, identifying coded speech is a matter of context and judgement, taking account of these guidelines.
5. Denying or minimizing the Holocaust by claiming that the deliberate Nazi genocide of the Jews did not take place, or that there were no extermination camps or gas chambers, or that the number of victims was a fraction of the actual total, is antisemitic.

B. Israel and Palestine: examples that, on the face of it, are antisemitic
6. Applying the symbols, images, and negative stereotypes of classical antisemitism (see guidelines 2 and 3) to the State of Israel.
7. Holding Jews collectively responsible for Israel's conduct or treating Jews, simply because they are Jewish, as agents of Israel.
8. Requiring people, because they are Jewish, publicly to condemn Israel or Zionism (for example, at a political meeting).
9. Assuming that non-Israeli Jews, simply because they are, Jews are necessarily more loyal to Israel than to their own countries.
10. Denying the right of Jews in the State of Israel to exist and flourish, collectively and individually, as Jews, in accordance with the principle of equality.

C. Israel and Palestine: examples that, on the face of it, are not antisemitic
(whether or not one approves of the view or action)
11. Supporting the Palestinian demand for justice and the full grant of their political, national, civil, and human rights, as encapsulated in international law.
12. Criticizing or opposing Zionism as a form of nationalism, or arguing for a variety of constitutional arrangements for Jews and Palestinians in the area between the Jordan River and the Mediterranean. It is not antisemitic to support arrangements that accord full equality to all inhabitants "between the river and the sea", whether in two states, a binational state, unitary democratic state, federal state, or in whatever form.
13. Evidence-based criticism of Israel as a state. This includes its institutions and founding principles. It also includes its policies and practices, domestic and abroad, such as the conduct of Israel in the West Bank and Gaza, the role Israel plays in the region, or any other way in which, as a state, it influences events in the world. It is not antisemitic to point out systematic racial discrimination. In general, the same norms of debate that apply to other states and to other conflicts over national self-determination apply in the case of Israel and Palestine. Thus, even if contentious, it is not antisemitic, in and of itself, to compare Israel with other historical cases, including settler-colonialism or apartheid.
14. Boycott, divestment, and sanctions are commonplace, non-violent forms of political protest against states. In the Israeli case they are not, in and of themselves, antisemitic.
15. Political speech does not have to be measured, proportional, tempered, or reasonable to be protected under article 19 of the Universal Declaration of Human Rights or article 10 of the European Convention on Human Rights and other human rights instruments. Criticism that some may see as excessive or contentious, or as reflecting a "double standard", is not, in and of itself, antisemitic. In general, the line between antisemitic and non-antisemitic speech is different from the line between unreasonable and reasonable speech.

Tweet: "{text}"

Is this tweet antisemitic according to the JDA definition? Provide a yes/no answer first, followed by a detailed explanation/justification for your decision. Format your response as: 
DECISION: [Yes/No]
EXPLANATION: [Your detailed explanation]
    """
}

class OllamaClusterClient:
    """Client for interacting with Ollama on a cluster."""
    
    def __init__(self, ip_port, model):
        """Initialize the client with the Ollama server IP:PORT and model name."""
        self.base_url = f"http://{ip_port}/api"
        self.model = model
        
    def pull_model(self):
        """Pull the model from Ollama if not already present."""
        url = f"{self.base_url}/pull"
        payload = {"model": self.model}
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"✅ Successfully pulled model {self.model}")
                return True
            else:
                print(f"❌ Failed to pull model: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error pulling model: {str(e)}")
            return False
    
    def generate(self, prompt, max_retries=3):
        """Send a generation request to Ollama."""
        url = f"{self.base_url}/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    # Parse the JSON response
                    response_json = response.json()
                    return response_json.get("response", "")
                else:
                    print(f"Error: {response.status_code}, {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                        continue
                    return f"ERROR: Status code {response.status_code}"
            except Exception as e:
                print(f"Exception: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return f"ERROR: {str(e)}"

def create_prompt(tweet_text, definition_type):
    """Create a prompt for the model using the specified definition template."""
    return DEFINITION_TEMPLATES[definition_type].format(text=tweet_text)

def extract_decision_explanation(response):
    """Extract the decision and explanation from the model's response."""
    try:
        # Try to find the formatted response
        if "DECISION:" in response and "EXPLANATION:" in response:
            decision_part = response.split("DECISION:")[1].split("EXPLANATION:")[0].strip()
            explanation_part = response.split("EXPLANATION:")[1].strip()
            
            # Clean up the decision to just get Yes/No
            decision = "Yes" if "yes" in decision_part.lower() else "No" if "no" in decision_part.lower() else "Unclear"
            
            return decision, explanation_part
        else:
            # For unformatted responses, try to extract a yes/no
            decision = "Yes" if "yes" in response.lower()[:50] else "No" if "no" in response.lower()[:50] else "Unclear"
            return decision, response
    except Exception as e:
        print(f"Error extracting decision: {str(e)}")
        return "Error", str(e)

def process_tweet(args):
    """Process a single tweet and return the results (for parallel processing)."""
    tweet_row, ollama_client = args
    tweet_id = tweet_row['TweetID']
    tweet_text = tweet_row['Text']
    
    result = {
        'TweetID': tweet_id,
        'Username': tweet_row['Username'],
        'CreateDate': tweet_row['CreateDate'],
        'Biased': tweet_row['Biased'],
        'Keyword': tweet_row['Keyword'],
        'Text': tweet_text
    }
    
    # Process with IHRA definition
    try:
        ihra_prompt = create_prompt(tweet_text, "IHRA")
        ihra_response = ollama_client.generate(ihra_prompt)
        ihra_decision, ihra_explanation = extract_decision_explanation(ihra_response)
        
        # Process with JDA definition
        jda_prompt = create_prompt(tweet_text, "JDA")
        jda_response = ollama_client.generate(jda_prompt)
        jda_decision, jda_explanation = extract_decision_explanation(jda_response)
        
        # Add results to the row
        result['IHRA_Decision'] = ihra_decision
        result['IHRA_Explanation'] = ihra_explanation
        result['JDA_Decision'] = jda_decision
        result['JDA_Explanation'] = jda_explanation
    except Exception as e:
        print(f"Error processing tweet {tweet_id}: {str(e)}")
        result['IHRA_Decision'] = "Error"
        result['IHRA_Explanation'] = str(e)
        result['JDA_Decision'] = "Error"
        result['JDA_Explanation'] = str(e)
    
    return result

def process_csv_cluster(input_file, output_file, ip_port, model_name, max_workers=MAX_WORKERS, batch_size=BATCH_SIZE, limit=None):
    """
    Process the CSV file using a cluster-hosted Ollama instance.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        ip_port: IP:PORT of the Ollama instance on the cluster
        model_name: Name of the model to use
        max_workers: Maximum number of parallel workers
        batch_size: Number of tweets to process before saving intermediate results
        limit: Optional limit on the number of rows to process (for testing)
    """
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Apply limit if specified
    if limit and limit > 0:
        df = df.head(limit)
        print(f"Limited to processing {limit} rows (for testing)")
    
    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    
    # Initialize the Ollama client
    ollama_client = OllamaClusterClient(ip_port, model_name)
    
    # Pull the model
    if not ollama_client.pull_model():
        print("Failed to pull the model. Please check the Ollama server and try again.")
        return
    
    # Create a results list and directory for intermediate results
    all_results = []
    os.makedirs("cluster_temp", exist_ok=True)
    
    # Process in batches
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_rows-1)//batch_size + 1} (tweets {batch_start+1}-{batch_end})")
        
        # Process batch in parallel
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for each tweet in the batch
            futures = [executor.submit(process_tweet, (row, ollama_client)) for _, row in batch_df.iterrows()]
            
            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tweets"):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error processing tweet: {str(e)}")
        
        # Add batch results to all results
        all_results.extend(batch_results)
        
        # Save intermediate results
        interim_df = pd.DataFrame(batch_results)
        # Extract model name from output_file path for better organization
        output_dir = os.path.dirname(output_file)
        batch_dir = os.path.join(output_dir, "batches")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Create a batch filename based on the output filename
        base_output = os.path.basename(output_file)
        base_name = os.path.splitext(base_output)[0]
        interim_file = f"{batch_dir}/{base_name}_batch_{batch_start+1}-{batch_end}.csv"
        
        interim_df.to_csv(interim_file, index=False)
        print(f"Saved interim results to {interim_file}")
        
        # Also update the full results file
        full_results_df = pd.DataFrame(all_results)
        full_results_df.to_csv(output_file, index=False)
        print(f"Updated full results in {output_file}")
    
    # Final save of all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    print(f"All results saved to {output_file}")
    
    # Generate some statistics
    ihra_yes_count = results_df[results_df['IHRA_Decision'] == 'Yes'].shape[0]
    ihra_no_count = results_df[results_df['IHRA_Decision'] == 'No'].shape[0]
    jda_yes_count = results_df[results_df['JDA_Decision'] == 'Yes'].shape[0]
    jda_no_count = results_df[results_df['JDA_Decision'] == 'No'].shape[0]
    
    print("\nFinal Statistics:")
    print(f"IHRA: {ihra_yes_count} antisemitic, {ihra_no_count} not antisemitic")
    print(f"JDA: {jda_yes_count} antisemitic, {jda_no_count} not antisemitic")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Analyze tweets for antisemitism using Ollama on a cluster')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, 
                        help='Output CSV file path')
    parser.add_argument('--ip-port', type=str, required=True, 
                        help='IP:PORT of the Ollama instance on the cluster')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, 
                        help='Ollama model name to use')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, 
                        help='Maximum number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                        help='Number of tweets to process before saving intermediate results')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of tweets to process (for testing)')
    
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file '{args.input}' not found")
        return
    
    # Process the CSV file
    process_csv_cluster(
        args.input, 
        args.output, 
        args.ip_port, 
        args.model, 
        args.workers, 
        args.batch_size, 
        args.limit
    )

if __name__ == "__main__":
    main()
