from graphviz import Digraph

def build_ctde_graph():
    g = Digraph("CTDE_Network", format="png")
    g.attr(rankdir="LR", splines="spline", nodesep="0.45", ranksep="0.6")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#F8FBFF", color="#5B7FA3", fontname="Arial")
    g.attr("edge", color="#4C6A8A")

    # ===== Actor =====
    with g.subgraph(name="cluster_actor") as a:
        a.attr(label="Actor（分布式决策）", color="#7AA6D8", style="rounded")

        a.node("S", "self_state [B,10]")
        a.node("N", "neighbors [B,Nn,10]")
        a.node("T", "tasks [B,Nt,20]")
        a.node("M", "mask [B,Nt]")

        a.node("Ls", "Linear 10→H")
        a.node("Ln", "Linear 10→H")
        a.node("Lt", "Linear 20→H")
        a.node("Dn", "Dropout")
        a.node("Dt", "Dropout")

        a.node("HS", "h_self [B,1,H]")
        a.node("HN", "h_neighbors [B,Nn,H]")
        a.node("AttnN", "MHA\nQ=h_self\nK/V=h_neighbors")
        a.node("HNA", "h_neighbor_agg [B,1,H]")
        a.node("HA", "h_agent = h_self + h_neighbor_agg")

        a.node("VC", "valid_counts=(~mask).sum")
        a.node("TC", "task_count_embed\n1→H/4→H")
        a.node("HAC", "h_agent_ctx = h_agent + 0.1*count_feat")

        a.node("PE", "Sinusoidal PE (+0.1)")
        a.node("HT", "h_tasks [B,Nt,H]")

        a.node("AttnT", "MHA\nQ=h_tasks\nK/V=h_agent_ctx")
        a.node("Dta", "Dropout")
        a.node("HF", "h_fused = h_tasks + h_task_attn")

        a.node("Head", "Linear H→1")
        a.node("Logits", "logits [B,Nt]")
        a.node("MaskFill", "masked_fill(mask, -1e9)")
        a.node("Dist", "Categorical(logits)")

        a.edges([
            ("S","Ls"), ("Ls","HS"),
            ("N","Ln"), ("Ln","Dn"), ("Dn","HN"),
            ("HS","AttnN"), ("HN","AttnN"), ("AttnN","HNA"),
            ("HS","HA"), ("HNA","HA"),
            ("M","VC"), ("VC","TC"), ("TC","HAC"), ("HA","HAC"),
            ("T","Lt"), ("Lt","Dt"), ("Dt","HT"), ("PE","HT"),
            ("HT","AttnT"), ("HAC","AttnT"), ("AttnT","Dta"),
            ("HT","HF"), ("Dta","HF"),
            ("HF","Head"), ("Head","Logits"), ("Logits","MaskFill"), ("M","MaskFill"),
            ("MaskFill","Dist")
        ])

    # ===== Critic =====
    with g.subgraph(name="cluster_critic") as c:
        c.attr(label="Critic（集中式评估）", color="#78B98B", style="rounded")

        c.node("GA", "global_agents [B,Na,13]")
        c.node("GT", "global_tasks [B,Nt,23]")
        c.node("APM", "agent_padding_mask (opt)")
        c.node("TPM", "task_padding_mask (opt)")

        c.node("CA", "Linear 13→H")
        c.node("CT", "Linear 23→H")
        c.node("HGA", "h_agents [B,Na,H]")
        c.node("HGT", "h_tasks [B,Nt,H]")

        c.node("CAT", "concat entities")
        c.node("AE", "all_entities [B,Na+Nt,H]")
        c.node("KPM", "build key_padding_mask")
        c.node("CMHA", "MHA(all_entities,\nkey_padding_mask)")
        c.node("HG", "h_global [B,Na+Nt,H]")
        c.node("POOL", "Masked Mean Pooling\n(or Mean if no mask)")
        c.node("VHEAD", "MLP H→64→1\n(ReLU+Dropout)")
        c.node("V", "value [B,1]")

        c.edges([
            ("GA","CA"), ("CA","HGA"),
            ("GT","CT"), ("CT","HGT"),
            ("HGA","CAT"), ("HGT","CAT"), ("CAT","AE"),
            ("APM","KPM"), ("TPM","KPM"),
            ("AE","CMHA"), ("KPM","CMHA"),
            ("CMHA","HG"), ("HG","POOL"), ("POOL","VHEAD"), ("VHEAD","V")
        ])

    return g

if __name__ == "__main__":
    g = build_ctde_graph()
    out = g.render(filename="ctde_network_architecture", cleanup=True)
    print(f"✅ 已生成: {out}")