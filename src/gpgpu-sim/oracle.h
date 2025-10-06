#ifndef ORACLE_PREFETCHER_H_
#define ORACLE_PREFETCHER_H_

#include <vector>
#include <deque>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstdint>

struct OracleDecodedAddr {
    unsigned chip;
    unsigned bk;
    unsigned row;
};

class OraclePrefetcher {
public:
    using addr_t = uint64_t;

    std::vector<uint8_t> claimed_;

    // Load the trace file (one address per line).
    // lookahead_win: how many subsequent requests to scan
    // max_pf: max number of prefetches per real request
    bool load(const char* path, unsigned lookahead_win = 20, unsigned max_pf = 2) {
        trace_.clear();
        pos_idx_.clear();
        lookahead_ = lookahead_win;
        max_pf_per_req_ = max_pf;

        std::ifstream fin(path);
        if (!fin) return enabled_ = false;

        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            // accept "0x..." (hex) or decimal
            unsigned long long v = 0ull;
            std::istringstream is(line);
            if (starts_with_(line, "0x") || starts_with_(line, "0X")) {
                is >> std::hex >> v;
            } else {
                is >> v;
            }
            addr_t a = static_cast<addr_t>(v);
            size_t idx = trace_.size();
            trace_.push_back(a);
            pos_idx_[a].push_back(idx);
        }

	claimed_.assign(trace_.size(), 0);

        return enabled_ = !trace_.empty();
    }

    bool enabled() const { return enabled_; }
    unsigned lookahead() const { return lookahead_; }
    unsigned max_pf_per_req() const { return max_pf_per_req_; }

    // Given a current real address `curr`, and the target (chip,bk,row),
    // return up to max_pf_per_req_ candidate addresses found within the next
    // `lookahead_` entries in the offline trace that map to the same (chip,bk,row).
    //
    // `addrdec_fn` is any callable: addr_t -> OracleDecodedAddr
    /*
    template <typename AddrDecoderFn>
    std::vector<addr_t> candidates(addr_t curr,
                                   unsigned want_chip,
                                   unsigned want_bk,
                                   unsigned want_row,
                                   AddrDecoderFn addrdec_fn)
    {
        std::vector<addr_t> out;
        if (!enabled_) return out;

        auto it = pos_idx_.find(curr);
        if (it == pos_idx_.end() || it->second.empty()) return out;

        // Consume the *next* occurrence to keep time monotonic.
        size_t pos = it->second.front();
        it->second.pop_front();

        const size_t end = std::min(trace_.size(), pos + 1 + lookahead_);
        for (size_t i = pos + 1; i < end; ++i) {
            addr_t cand = trace_[i];
            OracleDecodedAddr d = addrdec_fn(cand);
            if (d.chip == want_chip && d.bk == want_bk && d.row == want_row) {
                out.push_back(cand);
                if (out.size() >= max_pf_per_req_) break;
            }
        }
        return out;
    }
    */

template <typename AddrDecoderFn>
std::vector<addr_t> candidates(addr_t curr,
                               unsigned want_chip,
                               unsigned want_bk,
                               unsigned want_row,
                               AddrDecoderFn addrdec_fn)
{
    std::vector<addr_t> out;
    if (!enabled_) return out;

    auto it = pos_idx_.find(curr);
    if (it == pos_idx_.end() || it->second.empty()) return out;

    size_t pos = it->second.front();
    it->second.pop_front();

    const size_t end = std::min(trace_.size(), pos + 1 + lookahead_);
    if (end == 0 || end <= pos + 1) return out;

    for (size_t i = end - 1; i > pos; --i) {
        if (claimed_[i]) continue;              
        addr_t cand = trace_[i];
        OracleDecodedAddr d = addrdec_fn(cand);
        if (d.chip == want_chip && d.bk == want_bk && d.row == want_row) {
            out.push_back(cand);
            claimed_[i] = 1;                   
            if (out.size() >= max_pf_per_req_) break;
        }
    }
    return out;
}

private:
    static bool starts_with_(const std::string& s, const char* pfx) {
        const size_t n = std::char_traits<char>::length(pfx);
        return s.size() >= n && std::equal(pfx, pfx + n, s.begin());
    }

    std::vector<addr_t> trace_;
    std::unordered_map<addr_t, std::deque<size_t>> pos_idx_;
    unsigned lookahead_ = 20;
    unsigned max_pf_per_req_ = 2;
    bool enabled_ = false;
};

#endif // ORACLE_PREFETCHER_H_

