#ifndef ORACLE_PREFETCHER_H_
#define ORACLE_PREFETCHER_H_

#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct OracleDecodedAddr {
    unsigned chip;
    unsigned bk;
};

class OraclePrefetcher {
public:
    using addr_t = std::uint64_t;

    OraclePrefetcher() = default;

    bool load(const char* path, unsigned lookahead_win = 20) {
        reset_();
        std::ifstream fin(path);
        if (!fin) { enabled_ = false; return false; }
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            unsigned long long v = 0ull;
            std::istringstream is(line);
            if (starts_with_(line, "0x") || starts_with_(line, "0X")) is >> std::hex >> v;
            else is >> v;
            addr_t a = static_cast<addr_t>(v);
            const std::size_t idx = trace_.size();
            trace_.push_back(a);
            pos_idx_[a].push_back(idx);
        }
        claimed_.assign(trace_.size(), 0u);
        lookahead_ = lookahead_win;
        enabled_ = !trace_.empty();
        return enabled_;
    }

    bool enabled() const { return enabled_; }

    template <typename AddrDecoderFn>
    addr_t pick_next_addr_blp(addr_t curr,
                              unsigned want_chip,
                              const std::vector<unsigned>& bank_inflight,
                              AddrDecoderFn addrdec_fn)
    {
        if (!enabled_) return 0;
        auto it = pos_idx_.find(curr);
        if (it == pos_idx_.end() || it->second.empty()) return 0;
        const std::size_t pos = it->second.front();
        it->second.pop_front();
        if (bank_inflight.empty()) return 0;
        unsigned min_inflight = bank_inflight[0];
        for (std::size_t b = 1; b < bank_inflight.size(); ++b)
            if (bank_inflight[b] < min_inflight) min_inflight = bank_inflight[b];
        const std::size_t end = std::min(trace_.size(), pos + 1 + lookahead_);
        if (end == 0 || end <= pos + 1) return 0;
        for (std::size_t i = pos + 1; i < end; ++i) {
            if (claimed_[i]) continue;
            addr_t cand = trace_[i];
            OracleDecodedAddr d = addrdec_fn(cand);
            if (d.chip != want_chip) continue;
            if (d.bk >= bank_inflight.size()) continue;
            if (bank_inflight[d.bk] == min_inflight) {
                claimed_[i] = 1u;
                return cand;
            }
        }
        return 0;
    }

private:
    static bool starts_with_(const std::string& s, const char* pfx) {
        const std::size_t n = std::char_traits<char>::length(pfx);
        return s.size() >= n && std::equal(pfx, pfx + n, s.begin());
    }

    void reset_() {
        trace_.clear();
        pos_idx_.clear();
        claimed_.clear();
        enabled_ = false;
    }

    std::vector<addr_t> trace_;
    std::unordered_map<addr_t, std::deque<std::size_t>> pos_idx_;
    std::vector<std::uint8_t> claimed_;
    unsigned lookahead_ = 200;
    bool enabled_ = false;
};

#endif

