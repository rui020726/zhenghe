# 本轮全部远端尝试（实测逐条记录，含错误原文）

本文件是 `ROUND19_SCOPE.md` §一/§五 与 `CONSTRAINED_COMM_COST_PILOT.md` §七 的**追加件**：
那些文档写下时读数即正确，本文件只补后来的重试，**不改旧件**。
所有操作都是只读或只推送本会话自己的分支；**没有**注入凭据、**没有**改 git 配置、
**没有**改 hosts/解析、**没有**走本地代理（这些若要做，是使用者的决定，不是我的默认动作）。

## 一、尝试时间线

| # | 操作 | 结果（错误原文） |
|---|---|---|
| 1 | `git fetch origin refs/heads/docs/comm-cost-objective-plan:…` | `Recv failure: Connection was reset` |
| 2 | `git ls-remote origin refs/heads/docs/comm-cost-objective-plan` | `Failed to connect to github.com port 443 after 21091 ms` |
| 3 | 同上（第 2 次） | 同上（21 085 ms） |
| 4 | 同上（第 3 次） | 同上（21 065 ms） |
| 5 | `curl https://github.com` | `000`，`Failed to connect to github.com port 443 after 21040 ms` |
| 6 | `curl https://pypi.org`（对照） | **`200`** ⇒ 出站网络活着 |
| 7 | `curl https://api.github.com/repos/rui020726/zhenghe` | HTTP **404**（仓库私有，未认证读不到；**未尝试认证**） |
| 8 | `curl https://api.github.com/repos/…/git/ref/heads/docs/comm-cost-objective-plan` | HTTP **404**（同上） |
| 9 | `curl https://codeload.github.com` | HTTP 301（主机可达，但它不提供 git 端点） |
| 10 | `curl https://raw.githubusercontent.com` | TLS 握手失败 |
| 11 | `nslookup github.com` | 解析成功 → `20.205.243.166`（DNS 正常） |
| 12 | `git status`/`git log` 等本地操作 | 全部正常（本地不依赖网络） |
| — | `git push` | **未执行**：只读端点已确认不可达，先不做写入尝试 |
| 13 | （使用者要求重试）`git ls-remote origin refs/heads/docs/comm-cost-objective-plan` | `Recv failure: Connection was reset` |
| 14 | 同上 `git fetch …` | `Failed to connect to github.com port 443 after 21073 ms` |
| 15 | `git -c http.version=HTTP/1.1 ls-remote …`（排除 HTTP/2 与代理协商问题） | `Failed to connect to github.com port 443 after 21077 ms` |
| 16 | `netsh winhttp show proxy` | 「直接访问(没有代理服务器)」⇒ 系统未配代理 |
| 17 | `HKCU\…\Internet Settings` 的 `ProxyEnable`/`ProxyServer` 查询 | 无值 ⇒ 无浏览器级代理 |
| 18 | TCP 直连 `20.205.243.166:443` | **FAIL**；同时刻 `pypi.org` 仍 **200** |

**合计**：`git fetch` 2 次、`git ls-remote` 5 次（含 1 次 HTTP/1.1 变体）、`git push` 0 次；
**成功的远端接触数 = 0**。最后**实际核验过**的远端状态仍是
`refs/remotes/origin/analysis/real-search-undetected-attribution = 80af16ba4b90…`（第十七轮头）。

## 二、诊断结论（能排除的都排除了）

- **不是 git 的问题**：同一时刻 HTTP 层的 `pypi.org` 可达、`github.com` 的 **IP:443 TCP 直连失败**；
  DNS 解析正常；强制 HTTP/1.1 同样失败；系统没有配置任何代理。
- **不是认证问题**：失败发生在 TCP 建连，还没走到凭据。
- 因此这是**网络层对 `github.com:443` 的阻断**（解析到的 IP 与端口都不可达），
  不是重试次数或 git 选项能解决的。

## 三、可选的恢复路径（按侵入性从低到高，都等使用者点头）

1. **换出口重试**：手机热点 / 其它网络下执行 `CONSTRAINED_COMM_COST_PILOT.md` §七那六行；
   本轮结论与哈希只依赖本地树，晚推不影响正确性。
2. **在能连 GitHub 的机器上做一次离线搬运**：那边 `git bundle create plan.bundle
   refs/heads/docs/comm-cost-objective-plan`，把 bundle 交回本机 `git fetch plan.bundle`
   ⇒ 无需任何网络改动即可拿到 `72c4c3f71e07…` 的真实对象并核对，随后按本轮第 3 步原样
   cherry-pick（这条最符合"核对完整 SHA + 先验祖先关系再挑"的要求）。
3. **代理/VPN**：若使用者本机有可用代理，请你自己设定后我再跑 §七；我不会自行改 git 配置，
   也不会自行寻找代理端口。
4. 未读到那两份文档期间，本轮提案（基线、δ、资源上限、停止条件）**一律让位于它们**。
