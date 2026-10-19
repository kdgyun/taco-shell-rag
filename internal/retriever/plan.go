package retriever

import (
	"regexp"
	"strings"
)

var tokenPattern = regexp.MustCompile(`[\pL\pN_]+(?:[./:@-][\pL\pN_]+)*`)
var anchorPattern = regexp.MustCompile(`"([^"\n]{2,160})"|[A-Za-z][A-Za-z0-9]{1,15}-[0-9]+|[0-9]+\.[0-9]+(?:\.[0-9]+){0,2}|(?:[\pL\pN_.-]+/)+[\pL\pN_.-]+|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z]+(?:[A-Z][A-Za-z0-9]*)+`)

type Plan struct {
	Anchors        []string `json:"anchors"`
	Sources        []string `json:"sources"`
	OmittedAnchors int      `json:"omitted_anchors"`
	OmittedSources int      `json:"omitted_sources"`
	Strategy       string   `json:"strategy"`
}

func unique(values []string) []string {
	out, seen := []string{}, map[string]bool{}
	for _, value := range values {
		key := strings.ToLower(value)
		if !seen[key] {
			out = append(out, value)
			seen[key] = true
		}
	}
	return out
}

func queryTerms(query string) ([]string, bool) {
	terms := unique(tokenPattern.FindAllString(query, -1))
	truncated := len(terms) > 24
	if truncated {
		terms = terms[:24]
	}
	return terms, truncated
}

func makePlan(query string, sources []string, termLimit, sourceLimit int) Plan {
	terms := []string{}
	for _, match := range anchorPattern.FindAllStringSubmatch(query, -1) {
		value := match[0]
		if match[1] != "" {
			value = match[1]
		}
		terms = append(terms, value)
	}
	terms = unique(terms)
	p := Plan{Anchors: terms, Sources: sources, Strategy: "one query embedding; hybrid retrieval, exact query anchors and bounded source lanes"}
	if len(terms) > termLimit {
		p.OmittedAnchors = len(terms) - termLimit
		p.Anchors = terms[:termLimit]
	}
	if len(sources) > sourceLimit {
		p.OmittedSources = len(sources) - sourceLimit
		p.Sources = sources[:sourceLimit]
	}
	return p
}
