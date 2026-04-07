"""
Hãy viết một hàm Python để tính xác suất một email là Spam khi chứa một từ khóa nhất định.
"""

def bayes_spam_filter(p_spam, p_not_spam, p_word_if_spam, p_word_if_not_spam):
    # p_word = P(Word|Spam)*P(Spam) + P(Word|NotSpam)*P(NotSpam)
    p_word = p_word_if_spam * p_spam + p_word_if_not_spam * p_not_spam
    
    # p_spam_if_word = (P(Word|Spam) * P(Spam)) / P(Word)
    p_spam_if_word = (p_word_if_spam * p_spam) / p_word
    return p_spam_if_word

result = bayes_spam_filter(0.2, 0.8, 0.8, 0.1)
print(f"Xác suất là Spam: {result:.2%}")