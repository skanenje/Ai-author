def build_exchanges(turns):
    """
    Group turns into logical exchanges.
    Typically, an exchange is a user question followed by an assistant answer,
    or a sustained back-and-forth on a specific sub-topic.
    """
    exchanges = []
    current_exchange = []
    current_length = 0
    
    for turn in turns:
        text_len = len(turn.get('text', ''))
        # A simple heuristic: start a new exchange when the user speaks,
        # or if the roles are unknown and the current exchange is getting too large (~1500 chars)
        is_user = turn.get('role') == 'user'
        is_unknown_large = turn.get('role') == 'unknown' and current_length > 1500
        
        if (is_user or is_unknown_large) and current_exchange:
            exchanges.append(current_exchange)
            current_exchange = [turn]
            current_length = text_len
        else:
            current_exchange.append(turn)
            current_length += text_len
            
    if current_exchange:
        exchanges.append(current_exchange)
        
    return exchanges

def chunks_from_exchanges(exchanges):
    """
    Convert grouped exchanges into text chunks suitable for embedding.
    """
    chunks = []
    for idx, exchange in enumerate(exchanges):
        # Combine the text of all turns in the exchange
        text = "\n\n".join([f"{turn.get('role', 'unknown').capitalize()}: {turn.get('text', '')}" for turn in exchange])
        chunks.append({
            "id": idx,
            "text": text,
            "turn_count": len(exchange)
        })
    return chunks
