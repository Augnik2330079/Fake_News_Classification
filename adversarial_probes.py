import torch
import numpy as np

PROBE_ARTICLES = [
    {
        "id": "probe_1_formal_fake",
        "description": "Blatantly False Claim in Formal Reuters Wire Style",
        "factual_truth": "FALSE (Fabricated counterfactual)",
        "intended_style": "Real wire service (Reuters dateline, formal neutral tone)",
        "text": "WASHINGTON (Reuters) - The International Monetary Fund announced on Wednesday that the Moon's subterranean core consists predominantly of concentrated dairy proteins, prompting global central banks to adjust international reserve asset valuations. Treasury officials in London and Tokyo stated they would cooperate on market stability guidelines."
    },
    {
        "id": "probe_2_sensational_real",
        "description": "100% Verified Factual Event Written in Hyper-Sensational Clickbait Style",
        "factual_truth": "TRUE (Historical event: 2013 Chelyabinsk meteor explosion)",
        "intended_style": "Fake tabloid (Exclamation marks, ALL-CAPS, emotional hype)",
        "text": "OMG YOU WON'T BELIEVE THIS INSANE SHOCKER!! A MASSIVE METEOR JUST EXPLODED OVER RUSSIA IN A HUGE BLINDING FLASH, INJURING OVER 1,000 PEOPLE AND SHATTERING THOUSANDS OF WINDOWS! SCIENTISTS ARE TOTALLY STUNNED! WATCH THE SHOCKING FOOTAGE BEFORE IT GETS DELETED!!"
    },
    {
        "id": "probe_3_formal_pseudoscience",
        "description": "Fabricated Health Conspiracy in Calm Academic/Clinical Tone",
        "factual_truth": "FALSE (Debunked cellular conspiracy theory)",
        "intended_style": "Objective scientific / institutional prose",
        "text": "GENEVA (Reuters) - Epidemiological researchers from the European Telecommunications Standards Institute published preliminary laboratory findings indicating that millimeter-wave electromagnetic frequencies at 28 GHz trigger accelerated endothelial membrane degeneration in mammalian tissue samples."
    },
    {
        "id": "probe_4_satirical_formal",
        "description": "Satire/Absurdity in Professional Journalistic Deadpan Style",
        "factual_truth": "FALSE (Satirical parody)",
        "intended_style": "Formal political reportage",
        "text": "WASHINGTON (Reuters) - Congressional leaders reached a bipartisan accord on Friday to postpone the arrival of winter by three calendar weeks, citing infrastructure readiness backlogs and supply-chain pressures across the northeastern power grid."
    },
    {
        "id": "probe_5_plain_real_unadorned",
        "description": "Verified Fact Stated Casually Without Journalistic Credentials",
        "factual_truth": "TRUE (James Webb Space Telescope launch and primary mirror alignment)",
        "intended_style": "Informal / unadorned personal phrasing",
        "text": "Hey guys, NASA actually finished aligning the gold hexagonal mirrors on the James Webb Space Telescope out at the second Lagrange point, and the first deep field infrared images of distant galaxy clusters look completely crystal clear."
    }
]

def run_probes(lr_clf, tfidf_vec, distilbert_model, tokenizer, device):
    print("\n========================================================")
    print("  RUNNING ADVERSARIAL STRESS PROBES: STYLE VS. TRUTH    ")
    print("========================================================")
    
    probe_results = []
    
    distilbert_model.eval()
    
    for p in PROBE_ARTICLES:
        text = p["text"]
        
        # Logistic Regression Prediction
        X_vec = tfidf_vec.transform([text])
        lr_pred = int(lr_clf.predict(X_vec)[0])
        lr_prob_real = float(lr_clf.predict_proba(X_vec)[0, 1])
        lr_prob_fake = 1.0 - lr_prob_real
        
        # DistilBERT Prediction
        enc = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        with torch.no_grad():
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            logits = distilbert_model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            db_pred = int(np.argmax(probs))
            db_prob_fake = float(probs[0])
            db_prob_real = float(probs[1])
            
        res = {
            "id": p["id"],
            "description": p["description"],
            "factual_truth": p["factual_truth"],
            "text": text,
            "lr_prediction": "REAL (1)" if lr_pred == 1 else "FAKE (0)",
            "lr_confidence": f"{max(lr_prob_real, lr_prob_fake)*100:.1f}% ({'Real' if lr_pred == 1 else 'Fake'})",
            "lr_prob_real": lr_prob_real,
            "distilbert_prediction": "REAL (1)" if db_pred == 1 else "FAKE (0)",
            "distilbert_confidence": f"{max(db_prob_real, db_prob_fake)*100:.1f}% ({'Real' if db_pred == 1 else 'Fake'})",
            "distilbert_prob_real": db_prob_real
        }
        probe_results.append(res)
        
        print(f"\n[Probe: {p['id']}] - {p['description']}")
        print(f"  Factual Ground Truth: {p['factual_truth']}")
        print(f"  Text excerpt: \"{text[:110]}...\"")
        print(f"  -> Logistic Regression : Predicted {res['lr_prediction']} with confidence {res['lr_confidence']}")
        print(f"  -> DistilBERT          : Predicted {res['distilbert_prediction']} with confidence {res['distilbert_confidence']}")
        
        # Highlight discrepancy
        is_db_wrong = ("TRUE" in p["factual_truth"] and db_pred == 0) or ("FALSE" in p["factual_truth"] and db_pred == 1)
        if is_db_wrong:
            print(f"  *** CRITICAL FAILURE: Model predicted {res['distilbert_prediction']} based on linguistic STYLE, completely inverting FACTUAL TRUTH!")
            
    return probe_results
