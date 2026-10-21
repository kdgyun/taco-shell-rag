package core

import (
	"strings"
	"unicode"
)

func word(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_' }

func ContainsTerm(text, term string) bool {
	// FTS가 2.17을 2와 17로 나눠도 2.17.1 같은 다른 식별자를 확정 일치로 내보내지 않는다.
	// Recheck punctuation-bearing identifiers after FTS tokenization, including version suffixes.
	body, needle := []rune(strings.ToLower(text)), []rune(strings.ToLower(term))
	for i := 0; i+len(needle) <= len(body); i++ {
		if string(body[i:i+len(needle)]) != string(needle) {
			continue
		}
		before := i > 0 && word(body[i-1])
		end := i + len(needle)
		after := end < len(body) && word(body[end])
		if strings.ContainsAny(term, ".-/:") {
			before = before || (i > 1 && strings.ContainsRune(".-/:_", body[i-1]) && word(body[i-2]))
			after = after || (end+1 < len(body) && strings.ContainsRune(".-/:_", body[end]) && word(body[end+1]))
		}
		if !before && !after {
			return true
		}
	}
	return false
}
