package core

import "testing"

func TestLiteralTermBoundaries(t *testing.T) {
	for _, tc := range []struct {
		text, term string
		want       bool
	}{
		{"ABC-123 policy", "abc-123", true}, {"ABC-1234 policy", "ABC-123", false},
		{"version 2.17.0", "2.17", false}, {"version 2.17", "2.17", true},
		{"refund_policy", "refund", false}, {"환불 정책", "환불", true}, {"환불정책", "환불", false},
		{"docs/refunds.md", "docs/refunds.md", true}, {"docs/refunds.md.bak", "docs/refunds.md", false},
	} {
		if got := ContainsTerm(tc.text, tc.term); got != tc.want {
			t.Errorf("ContainsTerm(%q,%q)=%v", tc.text, tc.term, got)
		}
	}
}
