package logreg

type rng struct {
	state uint64
}

func randNew(seed uint64) *rng {
	return &rng{state: seed}
}

func (r *rng) Intn(n int) int {
	r.state ^= r.state << 13
	r.state ^= r.state >> 7
	r.state ^= r.state << 17
	return int(r.state % uint64(n))
}
