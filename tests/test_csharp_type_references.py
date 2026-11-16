"""C# members reference the types in their signatures.

``references_to`` and the impact radius read ``REFERENCES`` edges. For C# a
member that only *uses* a type (as a parameter, return, field or property type)
had no edge to it, so the type looked unused. These tests cover the extractor
(one position at a time), the namespace-aware binding in ``csharp_resolver``
(same lookup as constructor calls) and the graph lifecycle around it.
"""

import json
from pathlib import Path

import pytest
from tree_sitter_language_pack import get_language

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, incremental_update
from code_review_graph.parser import (
    _CSHARP_TYPE_SLOTS,
    _CSHARP_TYPE_WRAPPERS,
    CodeParser,
)
from code_review_graph.tools.query import query_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _store(root: Path) -> GraphStore:
    return GraphStore(root / ".code-review-graph" / "graph.db")


def _parse_refs(source: str) -> list[tuple[str, str]]:
    """``(member, type)`` pairs; the member is the source's name after ``::``.

    Several uses of one type on one line are one edge once stored (the store
    keys an edge by kind, endpoints and line), so they are one pair here too.
    """
    _, edges = CodeParser().parse_bytes(Path("S.cs"), source.encode())
    return sorted({
        (e.source.split("::", 1)[-1], e.target) for e in edges if e.kind == "REFERENCES"
    })


def _references(store: GraphStore, root: Path) -> list[tuple[str, str]]:
    prefix = f"{root}/"
    return sorted(
        (row[0].replace(prefix, ""), row[1].replace(prefix, ""))
        for row in store._conn.execute(
            "SELECT source_qualified, target_qualified FROM edges WHERE kind = 'REFERENCES'",
        )
    )


def _build(root: Path) -> list[tuple[str, str]]:
    with _store(root) as store:
        full_build(root, store)
        return _references(store, root)


_DOMAIN = "namespace Domain { public class Customer {} public class Order {} }"


# ---------------------------------------------------------------------------
# Extraction, one position at a time (single file)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("member", "expected"), [
    ("public void Save(Order o) {}", [("C.Save", "Order")]),
    ("public Order Get() => null;", [("C.Get", "Order")]),
    ("private Order _o;", [("C", "Order")]),
    ("public Order Owner { get; set; }", [("C", "Order")]),
    ("public C(Order o) {}", [("C.C", "Order")]),
    ("public List<Order> All() => null;", [("C.All", "List<Order>"), ("C.All", "Order")]),
    ("public Dictionary<string, List<Order>> M;", [
        ("C", "Dictionary<string,List<Order>>"), ("C", "List<Order>"), ("C", "Order"),
    ]),
    ("public Order[] Many() => null;", [("C.Many", "Order")]),
    ("public Order? Maybe() => null;", [("C.Maybe", "Order")]),
    ("public Order?[] Both;", [("C", "Order")]),
    ("public ref Order Ref(in Order a, out Order b) { b = null; return ref a; }", [
        ("C.Ref", "Order"),
    ]),
    ("public void P(params Order[] rest) {}", [("C.P", "Order")]),
    ("public (Order a, int b) Pair() => default;", [("C.Pair", "Order")]),
    ("public event EventHandler<Order> Changed;", [
        ("C", "EventHandler<Order>"), ("C", "Order"),
    ]),
    ("public Order this[Customer key] => null;", [("C", "Customer"), ("C", "Order")]),
    ("public delegate Order Make(Customer c);", [("C", "Customer"), ("C", "Order")]),
    ("public static C operator +(C a, Order b) => a;", [("C", "C"), ("C", "Order")]),
    ("public void Outer() { Order Local(Customer c) => null; }", [
        ("C.Outer", "Customer"), ("C.Outer", "Order"),
    ]),
    ("public void Q(A.B.Order o, global::X.Order g, M::Order m) {}", [
        ("C.Q", "A.B.Order"), ("C.Q", "M::Order"), ("C.Q", "global::X.Order"),
    ]),
    ("public Outer<Order>.Inner Q2;", [("C", "Order"), ("C", "Outer<Order>.Inner")]),
])
def test_signature_position_emits_a_reference(member, expected):
    assert _parse_refs(f"namespace N {{ class C {{ {member} }} }}") == sorted(
        (f"N.{source}", target) for source, target in expected
    )


def test_record_and_struct_primary_constructor_parameters_reference_their_types():
    assert _parse_refs(
        "namespace N { record R(Order O, int I); struct S(Customer C); "
        "class K(Order O) {} }"
    ) == [("N.K", "Order"), ("N.R", "Order"), ("N.S", "Customer")]


def test_interface_members_reference_their_types():
    assert _parse_refs(
        "interface IRepo { Order Get(int id); void Save(Order o, Customer c); "
        "Order Current { get; } }"
    ) == [
        ("IRepo", "Order"), ("IRepo.Get", "Order"),
        ("IRepo.Save", "Customer"), ("IRepo.Save", "Order"),
    ]


def test_namespace_level_delegate_is_attributed_to_the_file():
    _, edges = CodeParser().parse_bytes(
        Path("S.cs"), b"namespace N { delegate Order Make(Customer c); }",
    )
    assert sorted((e.source, e.target) for e in edges if e.kind == "REFERENCES") == [
        ("S.cs", "Customer"), ("S.cs", "Order"),
    ]


def test_edge_carries_the_evidence_the_binder_needs():
    _, edges = CodeParser().parse_bytes(
        Path("S.cs"),
        b"namespace N {\n  class C {\n    void M(Order o) {}\n  }\n}\n",
    )
    (edge,) = [e for e in edges if e.kind == "REFERENCES"]
    assert (edge.source, edge.target, edge.line) == ("S.cs::N.C.M", "Order", 3)
    scopes = edge.extra.pop("csharp_scopes")
    assert [name for name, _ in scopes] == ["N", ""] and scopes[-1][1] == -1
    assert edge.extra == {
        "csharp_raw_target": "Order",
        "csharp_containing_type": "N.C",
        "unresolved_targets": [],
    }


def test_a_type_used_twice_by_one_member_is_stored_as_one_edge(tmp_path):
    _write(tmp_path, {"Svc.cs": "class Svc { void M(Order a, Order b, List<Order> c) {} }"})
    assert _build(tmp_path) == [("Svc.cs::Svc.M", "List<Order>"), ("Svc.cs::Svc.M", "Order")]


@pytest.mark.parametrize("member", [
    "int A(string s, object o, bool b, long l, decimal d, nint n) => 0;",
    "void B(dynamic d) {}",
    "(int, string) D() => default;",
    "int[] E;",
    "int? F;",
    "int Prop { get; }",
])
def test_predefined_and_dynamic_types_emit_nothing(member):
    assert _parse_refs(f"class C {{ {member} }}") == []


def test_type_parameters_of_the_enclosing_class_and_method_are_skipped():
    assert _parse_refs(
        "class Box<T, U> { T Get<V>(V v, List<T> l, Box<T, U> g, Order o) => default; "
        "U Value; }"
    ) == [("Box.Get", "Box<T,U>"), ("Box.Get", "List<T>"), ("Box.Get", "Order")]


def test_variance_modifiers_do_not_hide_a_type_parameter():
    assert _parse_refs("interface I<out T> { T Next(); }") == []


@pytest.mark.parametrize("body", [
    "class D : Base, IFace {}",                      # base lists are INHERITS
    "class D { void M() { var x = new Order(); } }",  # object creation is a call
    "class D { void M(object o) { var x = (Order)o; } }",
    "class D { void M(object o) { var t = typeof(Order); } }",
    "class D { void M(object o) { if (o is Order p) {} } }",
    "class D { void M() { Order local = null; var y = local; } }",
    "class D { void M() { Func<int> f = () => 1; Action<Order> g = (Order x) => {}; } }",
    "[Serializable] class D {}",
    "class D<T> where T : Order {}",
])
def test_constructs_outside_signatures_emit_nothing(body):
    refs = [r for r in _parse_refs(body) if r[1] in ("Order", "Serializable", "Base", "IFace")]
    assert refs == []


def test_ts_style_dispatch_names_are_real_csharp_grammar_names():
    grammar = get_language("csharp")
    for node_type, field in _CSHARP_TYPE_SLOTS.items():
        kind = grammar.id_for_node_kind(node_type, True)
        assert kind is not None and grammar.node_kind_for_id(kind) == node_type, node_type
        assert grammar.node_kind_is_visible(kind), node_type
        assert grammar.field_id_for_name(field) is not None, field
    for node_type in (
        *_CSHARP_TYPE_WRAPPERS, "identifier", "qualified_name", "alias_qualified_name",
        "generic_name", "tuple_type", "tuple_element", "type_parameter_list",
        "type_parameter", "variable_declaration", "lambda_expression",
        "anonymous_method_expression",
    ):
        kind = grammar.id_for_node_kind(node_type, True)
        assert kind is not None and grammar.node_kind_for_id(kind) == node_type, node_type
        assert grammar.node_kind_is_visible(kind), node_type


# ---------------------------------------------------------------------------
# Cross-file binding
# ---------------------------------------------------------------------------


def test_using_namespace_binds_across_files(tmp_path):
    _write(tmp_path, {
        "Domain.cs": _DOMAIN,
        "Svc.cs": "using Domain; namespace App { class Svc { Order Find(Customer c) => null; } }",
    })
    assert _build(tmp_path) == [
        ("Svc.cs::App.Svc.Find", "Domain.cs::Domain.Customer"),
        ("Svc.cs::App.Svc.Find", "Domain.cs::Domain.Order"),
    ]


def test_resolved_edge_is_marked_inferred_like_a_bound_call(tmp_path):
    _write(tmp_path, {
        "Domain.cs": _DOMAIN, "Svc.cs": "using Domain; class Svc { Order O; }",
    })
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        (row,) = store._conn.execute(
            "SELECT * FROM edges WHERE kind = 'REFERENCES'",
        ).fetchall()
        extra = json.loads(row["extra"])
    assert row["confidence_tier"] == "INFERRED"
    assert row["target_resolution"] == "direct"
    assert extra["scoped_resolved"] is True
    assert "unresolved_targets" not in extra
    assert extra["csharp_raw_target"] == "Order"


def test_type_argument_binds_while_the_external_generic_stays_raw(tmp_path):
    _write(tmp_path, {
        "Domain.cs": _DOMAIN,
        "Svc.cs": "using Domain; using System.Collections.Generic; "
                  "class Svc { List<Order> All() => null; }",
    })
    assert _build(tmp_path) == [
        ("Svc.cs::Svc.All", "Domain.cs::Domain.Order"),
        ("Svc.cs::Svc.All", "List<Order>"),
    ]


def test_namespace_alias_and_type_alias_bind(tmp_path):
    _write(tmp_path, {
        "Domain.cs": _DOMAIN,
        "Svc.cs": "using M = Domain; using C = Domain.Customer; "
                  "class Svc { M.Order A() => null; C B; M::Order D; }",
    })
    assert _build(tmp_path) == [
        ("Svc.cs::Svc", "Domain.cs::Domain.Customer"),
        ("Svc.cs::Svc", "Domain.cs::Domain.Order"),
        ("Svc.cs::Svc.A", "Domain.cs::Domain.Order"),
    ]


def test_global_using_binds_inside_its_project_only(tmp_path):
    _write(tmp_path, {
        "Root.csproj": "<Project />",
        "Imports.cs": "global using Domain;",
        "Domain.cs": _DOMAIN,
        "Inside.cs": "class In { Order O; }",
        "Tests/Tests.csproj": "<Project />",
        "Tests/Outside.cs": "class Out { Order O; }",
    })
    assert _build(tmp_path) == [
        ("Inside.cs::In", "Domain.cs::Domain.Order"),
        ("Tests/Outside.cs::Out", "Order"),
    ]


def test_enclosing_namespace_and_fully_qualified_names_bind_without_a_using(tmp_path):
    _write(tmp_path, {
        "Domain.cs": "namespace Shop { public class Order {} } namespace Shop.Sub { class X {} }",
        "Sib.cs": "namespace Shop.Sub { class Y {\n Order O;\n X P;\n Shop.Order Q;\n"
                  " global::Shop.Order R;\n } }",
    })
    assert _build(tmp_path) == [
        ("Sib.cs::Shop.Sub.Y", "Domain.cs::Shop.Order"),
        ("Sib.cs::Shop.Sub.Y", "Domain.cs::Shop.Order"),
        ("Sib.cs::Shop.Sub.Y", "Domain.cs::Shop.Order"),
        ("Sib.cs::Shop.Sub.Y", "Domain.cs::Shop.Sub.X"),
    ]


def test_nested_types_bind_from_inside_and_from_outside(tmp_path):
    _write(tmp_path, {
        "Outer.cs": "namespace Lib { public class Outer { public class Inner {} "
                    "Inner Own() => null; } }",
        "Use.cs": "using Lib; class U { Outer.Inner A; Inner B; }",
    })
    refs = _build(tmp_path)
    assert ("Outer.cs::Lib.Outer.Own", "Outer.cs::Lib.Outer.Inner") in refs
    assert ("Use.cs::U", "Outer.cs::Lib.Outer.Inner") in refs
    # ``Inner`` alone is not in scope for ``U``: a bare name never reaches a nested type.
    assert ("Use.cs::U", "Inner") in refs


def test_generic_arity_never_selects_a_declaration_it_does_not_name(tmp_path):
    _write(tmp_path, {
        "Generic.cs": "namespace Lib; public class Box<T> {}",
        "Plain.cs": "namespace Lib; public class Box {}",
        "Use.cs": "using Lib; class U { Box<int> G; Box P; Box<int, int> Two; }",
    })
    refs = _build(tmp_path)
    assert ("Use.cs::U", "Generic.cs::Lib.Box") in refs
    assert ("Use.cs::U", "Plain.cs::Lib.Box") in refs
    # No two-parameter Box exists: the raw spelling stays, bound to nothing.
    assert ("Use.cs::U", "Box<int,int>") in refs
    assert len(refs) == 3


def test_generic_only_declaration_is_not_bound_by_a_plain_name(tmp_path):
    _write(tmp_path, {
        "Generic.cs": "namespace Lib; public class Box<T> {}",
        "Use.cs": "using Lib; class U { Box P; }",
    })
    assert _build(tmp_path) == [("Use.cs::U", "Box")]


def test_same_name_in_two_imported_namespaces_stays_unresolved(tmp_path):
    _write(tmp_path, {
        "A.cs": "namespace A { public class Item {} }",
        "B.cs": "namespace B { public class Item {} }",
        "Both.cs": "using A; using B; class Both { Item I; }",
        "OnlyA.cs": "using A; class OnlyA { Item I; }",
        "OnlyB.cs": "namespace B { class OnlyB { Item I; } }",
        "None.cs": "class None { Item I; }",
    })
    assert _build(tmp_path) == [
        # Never the wrong one: both candidates fit, so the name stays as written.
        ("Both.cs::Both", "Item"),
        ("None.cs::None", "Item"),
        ("OnlyA.cs::OnlyA", "A.cs::A.Item"),
        ("OnlyB.cs::B.OnlyB", "B.cs::B.Item"),
    ]


def test_ambiguous_edge_is_not_returned_by_references_to_for_either_type(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path, {
        "A.cs": "namespace A { public class Item {} }",
        "B.cs": "namespace B { public class Item {} }",
        "Both.cs": "using A; using B; class Both { Item I; }",
    })
    _build(tmp_path)
    for qualified in ("A.cs::A.Item", "B.cs::B.Item"):
        result = query_graph("references_to", str(tmp_path / qualified), str(tmp_path))
        assert result["status"] in ("ok", "not_found")
        assert result.get("results", []) == []


def test_external_type_creates_no_node_and_no_bound_edge(tmp_path):
    _write(tmp_path, {
        "Svc.cs": "using System; class Svc { Uri U; System.Uri V; Task<int> T() => null; }",
    })
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        refs = _references(store, tmp_path)
        names = {
            r[0] for r in store._conn.execute(
                "SELECT name FROM nodes WHERE language = 'csharp'",
            )
        }
        rows = store._conn.execute(
            "SELECT target_resolution, confidence_tier FROM edges WHERE kind = 'REFERENCES'",
        ).fetchall()
    assert refs == [
        ("Svc.cs::Svc", "System.Uri"), ("Svc.cs::Svc", "Uri"),
        ("Svc.cs::Svc.T", "Task<int>"),
    ]
    assert not names & {"Uri", "System.Uri", "Task", "Task<int>"}
    assert {tuple(r) for r in rows} == {("unresolved", "UNRESOLVED")}


def test_partial_type_declared_twice_stays_unresolved_rather_than_picking_a_part(tmp_path):
    _write(tmp_path, {
        "P1.cs": "namespace Lib; public partial class Form { }",
        "P2.cs": "namespace Lib; public partial class Form { }",
        "Use.cs": "using Lib; class U { Form F; }",
    })
    assert _build(tmp_path) == [("Use.cs::U", "Form")]


def test_type_shadowed_by_an_enclosing_type_binds_to_the_nearest(tmp_path):
    _write(tmp_path, {
        "Outer.cs": "namespace Lib { public class Order {} public class Outer { "
                    "public class Order {} Order Own() => null; } }",
    })
    assert _build(tmp_path) == [("Outer.cs::Lib.Outer.Own", "Outer.cs::Lib.Outer.Order")]


# ---------------------------------------------------------------------------
# Queries and impact
# ---------------------------------------------------------------------------


def test_references_to_lists_the_members_that_use_a_type(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path, {
        "Domain.cs": _DOMAIN,
        "Repo.cs": (
            "using Domain; namespace App { class Repo { "
            "Order Find(int id) => null; void Save(Order o, Customer c) {} "
            "Order _last; Customer Owner { get; } } "
            "class Unrelated { void Run(int x) {} } }"
        ),
    })
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
    result = query_graph("references_to", "Order", str(tmp_path))
    assert result["status"] == "ok", result.get("summary")
    assert sorted((r["name"], r["file_path"].rsplit("/", 1)[-1]) for r in result["results"]) == [
        ("Find", "Repo.cs"), ("Repo", "Repo.cs"), ("Save", "Repo.cs"),
    ]
    assert all(r["kind"] in ("Function", "Class") for r in result["results"])
    customer = query_graph("references_to", "Customer", str(tmp_path))
    assert sorted(r["name"] for r in customer["results"]) == ["Repo", "Save"]


def test_a_file_that_only_uses_a_type_is_in_the_impact_radius(tmp_path):
    _write(tmp_path, {
        "Domain.cs": _DOMAIN,
        "Repo.cs": "using Domain; class Repo { Order Find() => null; }",
        "Other.cs": "class Other { void Run() {} }",
    })
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        impact = store.get_impact_radius([str(tmp_path / "Domain.cs")])
    files = {Path(f).name for f in impact["impacted_files"]}
    assert "Repo.cs" in files
    assert "Other.cs" not in files


# ---------------------------------------------------------------------------
# Lifecycle: re-parse, rebinding, deletion, determinism
# ---------------------------------------------------------------------------


def test_reparsing_the_referrer_replaces_its_edges_without_duplicates(tmp_path):
    _write(tmp_path, {
        "Domain.cs": _DOMAIN, "Svc.cs": "using Domain; class Svc { Order O; }",
    })
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        (tmp_path / "Svc.cs").write_text(
            "using Domain; class Svc { Customer C; Order O; }", encoding="utf-8",
        )
        incremental_update(tmp_path, store, changed_files=["Svc.cs"])
        assert _references(store, tmp_path) == [
            ("Svc.cs::Svc", "Domain.cs::Domain.Customer"),
            ("Svc.cs::Svc", "Domain.cs::Domain.Order"),
        ]
        (tmp_path / "Svc.cs").write_text("using Domain; class Svc { }", encoding="utf-8")
        incremental_update(tmp_path, store, changed_files=["Svc.cs"])
        assert _references(store, tmp_path) == []


def test_incremental_update_matches_a_fresh_build(tmp_path):
    files = {
        "Domain.cs": _DOMAIN,
        "Svc.cs": "using Domain; class Svc { Order O; List<Customer> C; }",
    }
    _write(tmp_path / "inc", files)
    _write(tmp_path / "fresh", {**files, "Svc.cs": files["Svc.cs"] + " class Extra { Order E; }"})
    with _store(tmp_path / "inc") as store:
        full_build(tmp_path / "inc", store)
        (tmp_path / "inc" / "Svc.cs").write_text(
            files["Svc.cs"] + " class Extra { Order E; }", encoding="utf-8",
        )
        incremental_update(tmp_path / "inc", store, changed_files=["Svc.cs"])
        incremental = _references(store, tmp_path / "inc")
    assert incremental == _build(tmp_path / "fresh")


def test_namespace_rename_of_the_referenced_type_revokes_and_restores_binding(tmp_path):
    _write(tmp_path, {"Domain.cs": _DOMAIN, "Svc.cs": "using Domain; class Svc { Order O; }"})
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        (tmp_path / "Domain.cs").write_text(
            _DOMAIN.replace("namespace Domain", "namespace Renamed"), encoding="utf-8",
        )
        incremental_update(tmp_path, store, changed_files=["Domain.cs"])
        # The referrer was not re-parsed; it must not keep a binding to a
        # declaration that no longer exists.
        assert _references(store, tmp_path) == [("Svc.cs::Svc", "Order")]
        (tmp_path / "Domain.cs").write_text(_DOMAIN, encoding="utf-8")
        incremental_update(tmp_path, store, changed_files=["Domain.cs"])
        assert _references(store, tmp_path) == [("Svc.cs::Svc", "Domain.cs::Domain.Order")]


def test_deleting_the_referenced_file_keeps_the_raw_reference_and_rebinds_on_return(tmp_path):
    _write(tmp_path, {"Domain.cs": _DOMAIN, "Svc.cs": "using Domain; class Svc { Order O; }"})
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        (tmp_path / "Domain.cs").unlink()
        # A watch batch reports the deletion as a change and skips stale
        # reconciliation, so only the restore keeps the referrer's edge.
        incremental_update(
            tmp_path, store, changed_files=["Domain.cs"], reconcile_stale=False,
        )
        (row,) = store._conn.execute(
            "SELECT * FROM edges WHERE kind = 'REFERENCES'",
        ).fetchall()
        assert row["target_qualified"] == "Order"
        assert "unresolved_targets" in json.loads(row["extra"])
        (tmp_path / "Domain.cs").write_text(_DOMAIN, encoding="utf-8")
        incremental_update(tmp_path, store, changed_files=["Domain.cs"])
        assert _references(store, tmp_path) == [("Svc.cs::Svc", "Domain.cs::Domain.Order")]


def test_deleting_the_referenced_file_through_reconciliation_keeps_the_raw_reference(tmp_path):
    _write(tmp_path, {"Domain.cs": _DOMAIN, "Svc.cs": "using Domain; class Svc { Order O; }"})
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        (tmp_path / "Domain.cs").unlink()
        incremental_update(tmp_path, store)
        assert _references(store, tmp_path) == [("Svc.cs::Svc", "Order")]


def test_a_new_declaration_binds_references_in_unchanged_files(tmp_path):
    _write(tmp_path, {"Svc.cs": "using Domain; class Svc { Order O; }"})
    with _store(tmp_path) as store:
        full_build(tmp_path, store)
        assert _references(store, tmp_path) == [("Svc.cs::Svc", "Order")]
        _write(tmp_path, {"Domain.cs": _DOMAIN})
        incremental_update(tmp_path, store, changed_files=["Domain.cs"])
        assert _references(store, tmp_path) == [("Svc.cs::Svc", "Domain.cs::Domain.Order")]


def test_repeated_builds_give_identical_references(tmp_path):
    files = {
        "Domain.cs": _DOMAIN,
        "A.cs": "using Domain; class A { Order O; List<Customer> C; Dictionary<string, Order> D; }",
        "B.cs": "namespace Domain { class B { Customer C; Order Get(Customer c) => null; } }",
        "Dup1.cs": "namespace X { class Dup {} }",
        "Dup2.cs": "namespace Y { class Dup {} }",
        "Use.cs": "using X; using Y; class U { Dup D; }",
    }
    _write(tmp_path / "one", files)
    _write(tmp_path / "two", dict(reversed(list(files.items()))))
    assert _build(tmp_path / "one") == _build(tmp_path / "two")
    with _store(tmp_path / "one") as store:
        before = [tuple(r) for r in store._conn.execute(
            "SELECT kind, source_qualified, target_qualified, extra FROM edges "
            "WHERE kind = 'REFERENCES' ORDER BY source_qualified, target_qualified, line",
        )]
        full_build(tmp_path / "one", store)
        after = [tuple(r) for r in store._conn.execute(
            "SELECT kind, source_qualified, target_qualified, extra FROM edges "
            "WHERE kind = 'REFERENCES' ORDER BY source_qualified, target_qualified, line",
        )]
    assert before == after
