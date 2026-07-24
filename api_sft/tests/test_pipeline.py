from __future__ import annotations

import json
import collections
import inspect
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from api_sft.api_client import OpenAICompatibleClient, parse_json_object, resolve_api_key, user_message
from api_sft.catalogs import normalize_models, normalize_tools
from api_sft.common import load_env_file, load_yaml, sha256_file, stable_hash, write_json, write_jsonl
from api_sft.exporters import export_datasets
from api_sft.answers import generate_answers
from api_sft.questions import QUESTION_AUDIT_FORMAT_VERSION, QUESTION_CONTRACT_VERSION, QUESTION_RUNTIME_FORMAT_VERSION, QUESTION_RUNTIME_KEYS, UNIVERSAL_INGEST_TOOLS, _candidate_tool_pool, _model_catalog_scope, _requests_model_decision, _spec_compatible, _task_specs, assign_task_specs, build_question_specs, generate_questions, plan_modality_sampling, question_writer_payload, validate_question, write_coverage_report
from api_sft.scenarios import ARCHETYPES, TASK_RATIOS, WIDE_PANEL_LAYOUT, create_scenario, generate_scenarios, persist_scenario, rebuild_scenario_manifest, render_images, render_images_from_observed, to_wide_panel
from api_sft.signal_generator import DEFAULT_COMPLEXITY_MIX, complexity_schedule, render_signal, sample_signal_spec
from api_sft.verify import deterministic_review


ROOT=Path(__file__).resolve().parents[2]


class ScenarioTests(unittest.TestCase):
    def test_archetypes_cover_required_axes_and_are_reproducible(self):
        scenarios=[create_scenario(i+1,a,100+i) for i,a in enumerate(ARCHETYPES)]
        self.assertEqual({s.manifest["task"] for s in scenarios},{"data_profile","similarity_analysis","model_result_analysis","model_selection","tool_use"})
        self.assertEqual({s.manifest["series_count"] for s in scenarios},{"single","multiple"})
        self.assertEqual({s.manifest["history_length"] for s in scenarios},{"short","long"})
        self.assertEqual({s.manifest["recommended_mode"] for s in scenarios},{"text_only","text_then_image","image_text"})
        again=create_scenario(1,ARCHETYPES[0],100)
        self.assertEqual(scenarios[0].manifest["scenario_hash"],again.manifest["scenario_hash"])
        self.assertEqual(stable_hash(scenarios[0].truth),stable_hash(again.truth))

    def test_rendered_images_are_reproducible(self):
        scenario=create_scenario(1,ARCHETYPES[0],321)
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"a"; b=Path(tmp)/"b"; names_a=render_images(scenario,a); names_b=render_images(scenario,b)
            self.assertEqual(names_a,names_b)
            self.assertEqual([sha256_file(a/n) for n in names_a],[sha256_file(b/n) for n in names_b])

    def test_nonperiodic_archetypes_do_not_claim_a_period(self):
        for name in ["stationary_short_schema","short_model_selection","training_overfit","training_underfit","online_drift"]:
            archetype=next(a for a in ARCHETYPES if a["name"]==name)
            self.assertIsNone(create_scenario(1,archetype,99).truth["base_period"],name)

    def test_v4_complexity_mix_and_component_pool_coverage(self):
        levels=complexity_schedule(200,DEFAULT_COMPLEXITY_MIX,20260712)
        self.assertEqual(collections.Counter(levels),{"controlled":40,"compositional":120,"confounded":40})
        raw={task:200*ratio for task,ratio in TASK_RATIOS.items()}; quotas={task:int(value) for task,value in raw.items()}
        for task,_ in sorted(raw.items(),key=lambda item:item[1]-int(item[1]),reverse=True)[:200-sum(quotas.values())]: quotas[task]+=1
        by_task={task:[item for item in ARCHETYPES if item["task"]==task] for task in quotas}; schedule=[]
        while len(schedule)<200:
            for task in quotas:
                used=sum(item["task"]==task for item in schedule)
                if used<quotas[task]: schedule.append(by_task[task][used%len(by_task[task])])
        self.assertEqual({item["name"] for item in schedule},{item["name"] for item in ARCHETYPES})
        trends=set(); noises=set(); waveforms=set(); has_modulation=False; has_period_drift=False; missing_mechanisms=set(); general_tiers=set(); difficulty_image=set()
        for index,(archetype,complexity) in enumerate(zip(schedule,levels),1):
            scenario=create_scenario(index,archetype,20260712+index,complexity)
            self.assertEqual(scenario.manifest["generator_version"],"v4")
            self.assertIn(scenario.manifest["data_layout"],{"wide_panel_v1","training_curve_v1"})
            difficulty_image.add((scenario.manifest["analysis_difficulty"],scenario.manifest["image_value"]))
            signal=scenario.truth["generator"]["components"].get("signal")
            if signal:
                trends.add(signal["trend"]["type"]); noises.add(signal["noise"]["type"])
                for seasonal in signal["seasonality"]:
                    waveforms.add(seasonal["waveform"])
                    has_modulation |= bool(seasonal.get("amplitude_modulation"))
                    has_period_drift |= bool(seasonal.get("period_drift"))
            for event in scenario.truth["generator"]["events"].get("missingness",{}).values(): missing_mechanisms.add(event["mechanism"])
            multi=scenario.truth["generator"]["components"].get("multi_series")
            if multi and multi.get("realized_metrics"):
                self.assertTrue(multi["quality_gate"]["passed"])
                if multi["profile"]["type"]=="general":
                    general_tiers.add(multi["profile"]["tier"])
                    self.assertLessEqual(multi["realized_metrics"]["raw_pearson"]["mean"],.90)
                    self.assertFalse(multi["realized_metrics"]["all_pairs_above_0_90"])
                if archetype["pattern"]!="intermittent": self.assertTrue(multi["group_factors"]); self.assertTrue(multi["individual_factors"])
        self.assertEqual(trends,{"none","linear","quadratic","saturating","piecewise","local_reversal","random_walk"})
        self.assertEqual(noises,{"gaussian","student_t","ar1","heteroscedastic","level_dependent","mixture"})
        self.assertEqual(waveforms,{"sine","triangle","soft_square"})
        self.assertTrue(has_modulation); self.assertTrue(has_period_drift)
        self.assertEqual(general_tiers,{"low","medium","high"})
        self.assertIn(("medium","high"),difficulty_image); self.assertIn(("hard","medium"),difficulty_image)
        self.assertTrue(any("random" in item for item in missing_mechanisms)); self.assertTrue(any("synchronous" in item for item in missing_mechanisms)); self.assertTrue(any("value_dependent" in item for item in missing_mechanisms))

    def test_component_equation_reconstructs_observed_signal(self):
        sampling_rng=np.random.default_rng(41)
        spec=sample_signal_spec(sampling_rng,360,"confounded",seasonal_count=3)
        observed,parts=render_signal(np.random.default_rng(73),360,spec,scale=1.4,offset=-2.5,phase_offset=.3,idiosyncratic_scale=.8)
        np.testing.assert_allclose(observed,parts["clean"]+parts["noise"],rtol=0,atol=1e-12)
        again,_=render_signal(np.random.default_rng(73),360,spec,scale=1.4,offset=-2.5,phase_offset=.3,idiosyncratic_scale=.8)
        np.testing.assert_array_equal(observed,again)

    def test_visible_context_and_images_do_not_use_oracle_truth(self):
        archetype=next(item for item in ARCHETYPES if item["name"]=="point_anomaly_changepoint")
        scenario=create_scenario(1,archetype,777,"confounded")
        visible=json.dumps(scenario.manifest["visible_context"],ensure_ascii=False)
        for forbidden in ["ground_truth","generator","anomaly_indices","change_point","seasonal_periods","drift_type"]:
            self.assertNotIn(forbidden,visible)
        signature=inspect.signature(render_images_from_observed)
        self.assertEqual(list(signature.parameters),["frame","visible_context","out"])
        source=inspect.getsource(render_images_from_observed)
        self.assertNotIn("scenario.truth",source); self.assertNotIn("ground_truth",source)
        with tempfile.TemporaryDirectory() as tmp:
            record=persist_scenario(scenario,Path(tmp)); provenance=json.loads(Path(record["figure_provenance_path"]).read_text(encoding="utf-8"))
            self.assertTrue(provenance)
            self.assertTrue(all(item["oracle_truth_used"] is False for item in provenance.values()))
            self.assertTrue(all("method" in item and "input_columns" in item and "parameters" in item for item in provenance.values()))

    def test_default_50_replaces_degenerate_missingness_panel(self):
        count=50; seed=20260712; raw={task:count*ratio for task,ratio in TASK_RATIOS.items()}; quotas={task:int(value) for task,value in raw.items()}
        for task,_ in sorted(raw.items(),key=lambda item:item[1]-int(item[1]),reverse=True)[:count-sum(quotas.values())]: quotas[task]+=1
        by_task={task:[item for item in ARCHETYPES if item["task"]==task] for task in quotas}; schedule=[]
        while len(schedule)<count:
            for task in quotas:
                used=sum(item["task"]==task for item in schedule)
                if used<quotas[task]: schedule.append(by_task[task][used%len(by_task[task])])
        complexity=complexity_schedule(count,DEFAULT_COMPLEXITY_MIX,seed)[49]
        scenario=create_scenario(50,schedule[49],seed+50,complexity); multi=scenario.truth["generator"]["components"]["multi_series"]; metrics=multi["realized_metrics"]
        self.assertEqual(scenario.manifest["archetype"],"missingness_profile")
        self.assertEqual(scenario.manifest["analysis_difficulty"],"medium")
        self.assertLessEqual(metrics["raw_pearson"]["mean"],.90)
        transformations={(item["scale"],item["offset"],item["amplitude"]) for item in multi["series_transformations"].values()}
        self.assertGreater(len(transformations),1)
        groups=multi["group_assignment"]
        self.assertGreater(len(set(groups.values())),1); self.assertTrue(multi["group_factors"])

    def test_multiseries_retry_is_reproducible_and_exhaustion_fails(self):
        archetype=next(item for item in ARCHETYPES if item["name"]=="distribution_shape_mismatch")
        first=create_scenario(8,archetype,20260720,"confounded"); second=create_scenario(8,archetype,20260720,"confounded")
        self.assertEqual(first.manifest["scenario_hash"],second.manifest["scenario_hash"])
        self.assertEqual(first.truth["generator"]["components"]["multi_series"]["quality_gate"]["attempt"],2)
        with patch("api_sft.scenarios.relation_passes",return_value=(False,["forced_failure"])) as gate:
            with self.assertRaisesRegex(RuntimeError,"after 2 attempts"):
                create_scenario(8,archetype,20260720,"confounded",{"quality_gate":True,"max_attempts":2})
        self.assertEqual(gate.call_count,2)

    def test_missingness_figures_preserve_gaps_and_pairwise_provenance(self):
        archetype=next(item for item in ARCHETYPES if item["name"]=="missingness_profile")
        scenario=create_scenario(50,archetype,20260762,"compositional")
        with tempfile.TemporaryDirectory() as tmp:
            record=persist_scenario(scenario,Path(tmp)); provenance=json.loads(Path(record["figure_provenance_path"]).read_text())
        self.assertEqual(provenance["normalized_overlay.png"]["parameters"]["missing_handling"],"preserve_nan_gaps")
        self.assertEqual(provenance["correlation_heatmap.png"]["method"],"observed_pairwise_complete_pearson")
        self.assertIn("pairwise_overlap",provenance["correlation_heatmap.png"]["parameters"])
        self.assertEqual(provenance["dtw_alignment.png"]["parameters"]["missing_handling"],"linear_interpolation_then_edge_fill")

    def test_resume_recovers_directory_manifest_over_stale_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=generate_scenarios(1,20260712,root)
            stale=dict(rows[0]); stale["generator_version"]="v2"
            write_jsonl(root/"scenarios.jsonl",[stale])
            with patch("api_sft.scenarios.persist_scenario",wraps=persist_scenario) as persist:
                regenerated=generate_scenarios(1,20260712,root,resume=True)
            self.assertEqual(persist.call_count,0)
            self.assertEqual(regenerated[0]["generator_version"],"v4")

    def test_wide_panel_preserves_duplicate_rows_without_aggregation(self):
        source=pd.DataFrame({"time":[1,1,1,1,2,2],"series_id":["a","a","b","b","a","b"],"value":[10,11,20,21,12,22],"prediction":[9,10,19,20,11,21]})
        wide,mapping=to_wide_panel(source)
        self.assertEqual(list(wide.columns),["time","s01","s02","s01__prediction","s02__prediction"])
        self.assertEqual(int(source.duplicated(["time","series_id"]).sum()),2)
        self.assertEqual(int(wide.duplicated(["time"]).sum()),2)
        self.assertEqual(mapping["duplicate_policy"],"sparse_repeated_time_rows_no_aggregation")
        self.assertNotIn("occurrence",wide.columns)

    def test_v4_covers_all_persisted_schema_families(self):
        expected={
            "trend_period_profile":{"time","s01"},
            "residual_autocorrelation":{"time","s01","s01__prediction","s01__residual","s01__lower_90","s01__upper_90"},
            "online_drift":{"time","s01__absolute_error","s01__missing_rate","s01__feature_psi"},
            "training_overfit":{"step","train_loss","validation_loss"},
            "covariate_model_selection":{"time","s01","s01__promo_flag","s01__temperature","s01__actual_delivery_delay"},
            "hierarchical_model_selection":{"time","s01","s01__region","s01__store"},
            "leakage_metric_rules":{"time","s01","s01__target_t_plus_1","s01__future_window_target_mean"},
        }
        for index,(name,columns) in enumerate(expected.items(),1):
            archetype=next(item for item in ARCHETYPES if item["name"]==name)
            scenario=create_scenario(index,archetype,900+index,"compositional")
            self.assertTrue(columns<=set(scenario.frame.columns),name)
            self.assertNotIn("series_id",scenario.frame.columns,name)

    def test_generation_failure_checkpoints_completed_prefix(self):
        scenario=create_scenario(1,ARCHETYPES[0],20260713,"controlled")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch("api_sft.scenarios.create_scenario",side_effect=[scenario,RuntimeError("forced failure")]):
                with self.assertRaisesRegex(RuntimeError,"forced failure"):
                    generate_scenarios(2,20260712,root)
            checkpoint=list(map(json.loads,(root/"scenarios.jsonl").read_text(encoding="utf-8").splitlines()))
            self.assertEqual([row["id"] for row in checkpoint],["scenario_00001"])
            self.assertEqual(rebuild_scenario_manifest(root),checkpoint)


class CatalogTests(unittest.TestCase):
    def test_real_catalogs(self):
        bundle=Path("/Users/monychen/Documents/tsa/claude_tsa/tool_dataset_sources/data_analysis_tools_bundle.json")
        if not bundle.exists(): self.skipTest("External tools bundle not available")
        self.assertEqual(normalize_tools(bundle)["tool_count"],33)
        paths=[ROOT/"agent_tools"/f"metadata_{i}.json" for i in [1,2,3]]
        models=normalize_models(paths)
        self.assertGreater(models["model_count"],20)
        self.assertEqual(models["model_count"],len({f"{m['package']}::{m['name']}" for m in models["models"]}))


class QuestionTests(unittest.TestCase):
    def test_simple_question_is_rejected(self):
        self.assertFalse(validate_question("这个序列有没有趋势？")["passed"])

    def test_model_catalog_scope_is_only_coarse_task_class(self):
        self.assertEqual(_model_catalog_scope({"parent_task":"model_selection","id":"selection_history_length"},"forecast"),"forecast")
        self.assertEqual(_model_catalog_scope({"parent_task":"model_selection","id":"selection_anomaly_task"},"sequence_anomaly_detection"),"anomaly_detection")
        self.assertEqual(_model_catalog_scope({"parent_task":"data_profile","id":"profile_trend_types"},"forecast"),"none")

    def test_none_scope_rejects_new_model_selection_but_allows_existing_ab_comparison(self):
        self.assertTrue(_requests_model_decision("请检查时间索引，并选择可解释性更强的预测方法。"))
        self.assertFalse(_requests_model_decision("请比较已有模型A与模型B的残差，并决定继续使用A还是切换到B。"))
        spec=self._question_spec("q1","text_only","q1")
        metadata={"decision_points":["检查时间索引","选择预测方法"],"constraint_key":"interpretability","facts_used":["schema.columns"]}
        quality=validate_question("请判断时间索引是否可靠，并选择可解释性更强的预测方法，同时说明还需要补充哪些证据。",spec,metadata)
        self.assertFalse(quality["checks"]["model_scope"])
        self.assertTrue(quality["details"]["model_decision_requested"])

    def test_candidate_tool_pool_has_universal_tools_and_deterministic_distractors(self):
        tool_map={f"tool_{index}":{"name":f"tool_{index}"} for index in range(30)}
        for name in UNIVERSAL_INGEST_TOOLS: tool_map[name]={"name":name}
        spec={"preferred_tools":["tool_1","tool_2"]}
        first=_candidate_tool_pool(spec,"model_result_analysis",tool_map,42,"q1")
        second=_candidate_tool_pool(spec,"model_result_analysis",tool_map,42,"q1")
        names=[item["name"] for item in first]
        self.assertEqual(first,second); self.assertEqual(len(first),20)
        self.assertTrue(set(UNIVERSAL_INGEST_TOOLS)<=set(names)); self.assertTrue({"tool_1","tool_2"}<=set(names))

    def test_coverage_detects_missing_axes(self):
        base={"task":"data_profile","series_count":"single","history_length":"short","task_goal":"forecast","input_mode":"text_only","recommended_mode":"text_only","image_value":"low","difficulty":"medium","question":"x","question_quality":{"complexity_score":1}}
        with tempfile.TemporaryDirectory() as tmp:
            report=write_coverage_report([base],Path(tmp)/"coverage.json")
        self.assertFalse(report["passed"]); self.assertIn("task",report["missing_required_values"])

    def test_task_pool_covers_all_subtasks_and_tools(self):
        specs=_task_specs(ROOT/"api_sft"/"task_pool.yaml")
        self.assertEqual(len(specs),58)
        self.assertEqual({spec["task_difficulty"] for spec in specs},{"medium","hard"})
        bundle=Path("/Users/monychen/Documents/tsa/claude_tsa/tool_dataset_sources/data_analysis_tools_bundle.json")
        if not bundle.exists(): self.skipTest("External tools bundle not available")
        tools={x["name"] for x in normalize_tools(bundle)["tools"]}
        self.assertEqual({name for spec in specs for name in spec["preferred_tools"]},tools)
        quotas={"data_profile":50,"similarity_analysis":36,"model_result_analysis":40,"model_selection":44,"tool_use":30}; scenarios=[]; index=0
        for task,count in quotas.items():
            archetypes=[a for a in ARCHETYPES if a["task"]==task]
            for i in range(count):
                a=archetypes[i%len(archetypes)]; index+=1; scenarios.append({"id":f"s{index}","task":task,"pattern":a["pattern"],"archetype":a["name"]})
        assigned=assign_task_specs(scenarios,specs)
        self.assertEqual({spec["id"] for _,spec in assigned},{spec["id"] for spec in specs})
        self.assertTrue(all(_spec_compatible(spec,scenario) for scenario,spec in assigned))
        config={"ensure_each_subtask_both_modalities":False,"ensure_each_parent_task_both_modalities":True,"policies":{"low":{"text_only":.75,"image_text":.10,"paired":.15},"medium":{"text_only":.30,"image_text":.30,"paired":.40},"high":{"text_only":.10,"image_text":.75,"paired":.15}}}
        plans=plan_modality_sampling(assigned,config,20260712)
        high=[modes for _,spec,modes,_ in plans if spec.get("image_policy")=="high"]
        low=[modes for _,spec,modes,_ in plans if spec.get("image_policy")=="low"]
        self.assertGreater(sum("image_text" in modes for modes in high),sum(modes==["text_only"] for modes in high))
        self.assertGreater(sum("text_only" in modes for modes in low),sum(modes==["image_text"] for modes in low))
        strict={**config,"ensure_each_subtask_both_modalities":True}; strict_plans=plan_modality_sampling(assigned,strict,20260712); observed=collections.defaultdict(set)
        for _,spec,modes,_ in strict_plans: observed[spec["id"]].update(modes)
        self.assertTrue(all(modes=={"text_only","image_text"} for modes in observed.values()))

    def _question_spec(self, row_id: str, mode: str, pair_id: str) -> dict:
        return {
            "id":row_id,"question_spec_version":"4.0","question_spec_hash":stable_hash({"id":row_id,"mode":mode,"scenario":"hash"}),"question_group_id":pair_id,"scenario_id":"scenario_x","pair_id":pair_id,"is_paired":True,"pair_role":mode,"split_group":"scenario_x",
            "task":"data_profile","task_label":"数据画像","subtask_id":"profile_schema_frequency_index","subtask_title":"字段、频率与时间索引质量检查","task_goal":"forecast",
            "series_count":"single","history_length":"long","difficulty":"medium","input_mode":mode,"data_path":"/tmp/data.csv","dataset_attachment":{"path":"/tmp/data.csv","format":"csv","source_type":"local_path"},"images":["/tmp/x.png"] if mode=="image_text" else [],"image_inventory":["overview.png"] if mode=="image_text" else [],"image_attachments":[{"path":"/tmp/x.png","filename":"x.png","media_type":"image/png","source_type":"local_path"}] if mode=="image_text" else [],
            "evidence_packet":{"schema":{"columns":["time","s01"]},"data_scale":{"row_count":803,"series_count":1,"history_length_per_series":803},"time_index":{"frequency":"synthetic_step","observed_range":{"start":"0","end":"802"}},"statistics":{"s01":{"mean":1.2}},"business_constraints":{"compute_budget":"low","interpretability":"required","error_cost":"under_forecast_higher"},"known_future_covariates":[]},
            "visible_context":{},"recommended_mode":"text_only","image_value":"low","visual_reason":"x","recommended_plots":[],"text_can_answer":[],"image_should_answer":[],"requires_statistical_confirmation":[],
            "model_catalog_scope":"none","system_prompt_id":"tsa_tool_execution_v2","system_prompt":"TOOL SYSTEM","message_format":"neutral_local_images_v1","trajectory_requirement":"tool_execution","candidate_tools":[],"primary_tools":["data_profile"],"required_answer_elements":["schema判断"],
            "internal_rubric":{"task_instruction":"internal-only","required_elements":["schema判断"],"preferred_tools":["data_profile"],"image_policy":"low","expected_decision_points":2,"allowed_user_tool_mentions":[],"evidence_boundaries":[]},"scenario_hash":"hash",
        }

    def test_writer_payload_excludes_private_rubric_and_routing_labels(self):
        payload=question_writer_payload([self._question_spec("q1","text_only","pair")])
        raw=json.dumps(payload,ensure_ascii=False)
        for forbidden in ["internal_rubric","preferred_tools","image_policy","recommended_mode","candidate_models","ground_truth","visible_evidence","statistics","data_scale","observed_range","series_count","history_length","input_modes","available_image_types","difficulty","fine_grained_task"]:
            self.assertNotIn(forbidden,raw)
        self.assertIn("investigation_topic",payload["task"])
        self.assertIn("结果未知",payload["task"]["investigation_topic"])
        self.assertEqual(payload["required_output"]["required_decision_count"],2)
        self.assertIn("decision_constraints",payload["external_context"])
        self.assertIn("resource_semantics",payload["external_context"])
        self.assertIn("forbidden",payload["decision_boundaries"]["model_selection"])

    def test_writer_payload_is_safe_across_task_goals(self):
        cases=[
            ("model_selection","forecast","selection_history_length","known_future_covariates"),
            ("data_profile","diagnosis","profile_schema_frequency_index",None),
            ("similarity_analysis","sequence_anomaly_detection","similarity_anomalous_series","comparison_purpose"),
            ("model_result_analysis","monitoring_retraining","result_data_concept_drift","decision_objective"),
            ("tool_use","point_anomaly_detection","tool_anomaly_change_order","detection_granularity"),
        ]
        for task,goal,subtask,expected_context_key in cases:
            spec=self._question_spec(f"q_{task}","text_only",f"g_{task}")
            spec.update({"task":task,"task_label":task,"task_goal":goal,"subtask_id":subtask,"subtask_title":subtask})
            payload=question_writer_payload([spec]); raw=json.dumps(payload,ensure_ascii=False)
            for forbidden in ["statistics","data_scale","series_count","history_length","missing_ratio","observed_range","\"mean\"","\"std\""]:
                self.assertNotIn(forbidden,raw,(task,goal))
            self.assertIn("decision_constraints",raw)
            self.assertIn("resource_semantics",raw)
            if expected_context_key:
                self.assertIn(expected_context_key,payload["external_context"].get("task_specific_context",{}),(task,goal))
            if goal in {"point_anomaly_detection","sequence_anomaly_detection"}:
                self.assertNotIn("under_forecast_higher",raw)

    def test_question_resume_revalidates_old_contract_and_rewrites_scope_conflict(self):
        replacement={"user_request":"在可解释性约束下，请判断数据的字段角色和时间索引是否适合后续分析，并决定还需检查哪些证据来排除不规则间隔风险，同时说明何时需要转换格式。","decision_points":["判断索引是否适用","决定检查证据与转换条件"],"constraint_key":"interpretability","business_facts_used":["decision_constraints.interpretability"]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); specs=root/"specs.jsonl"; output=root/"questions.jsonl"; rejected=root/"rejected.jsonl"; prompt=root/"writer.txt"; prompt.write_text("只输出JSON")
            spec=self._question_spec("q1","text_only","q1"); write_jsonl(specs,[spec])
            old={"id":"q1","question_spec_hash":spec["question_spec_hash"],"user_request":"请检查时间索引，并选择一个可解释的预测模型，同时说明如何评估。","question_generation":{"decision_points":["检查索引","选择模型"],"constraint_key":"interpretability","facts_used":["schema.columns"]}}
            write_jsonl(output,[old])
            with patch("api_sft.questions.OpenAICompatibleClient.complete",return_value=(json.dumps(replacement,ensure_ascii=False),{},.1)) as complete:
                rows=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt,resume=True)
        self.assertEqual(complete.call_count,1)
        self.assertEqual(rows[0]["format_version"],QUESTION_RUNTIME_FORMAT_VERSION)
        self.assertNotIn("选择一个可解释的预测模型",rows[0]["prompt"]["user_request"])

    def test_direct_writer_keeps_paired_question_text_identical(self):
        response={"user_request":"我们准备把这批序列接入日常预测。在低算力约束下，请判断字段角色和时间索引是否适合安全建模，并决定需要检查哪些证据以排除泄漏或不规则间隔风险，同时说明何时才需要转换格式。","decision_points":["判断数据是否适用","决定检查证据与转换条件"],"constraint_key":"compute_budget","business_facts_used":["decision_constraints.compute_budget","resource_semantics"]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); specs=root/"specs.jsonl"; output=root/"questions.jsonl"; rejected=root/"rejected.jsonl"; prompt=root/"writer.txt"
            write_jsonl(specs,[self._question_spec("q1","text_only","pair"),self._question_spec("q2","image_text","pair")]); prompt.write_text("只输出JSON",encoding="utf-8")
            with patch("api_sft.questions.OpenAICompatibleClient.complete",return_value=(json.dumps(response,ensure_ascii=False),{"total_tokens":10},.1)) as complete:
                rows=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt,max_attempts=2)
            audits=[json.loads(line) for line in (root/"questions.audit.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows),2); self.assertEqual(rows[0]["prompt"]["user_request"],rows[1]["prompt"]["user_request"])
        self.assertEqual(complete.call_count,1)
        self.assertEqual(set(rows[0]),QUESTION_RUNTIME_KEYS)
        self.assertEqual(set(rows[0]["spec_ref"]),{"version","hash"})
        self.assertEqual(set(rows[0]["task"]),{"category","subtask_id","goal","input_mode","model_catalog_scope"})
        self.assertEqual(set(rows[0]["prompt"]),{"system_prompt_id","user_request"})
        self.assertEqual(set(rows[0]["resources"]),{"dataset","images"})
        self.assertEqual(set(rows[0]["resources"]["dataset"]),{"path","format"})
        self.assertLess(sum(len(json.dumps(row,ensure_ascii=False)) for row in rows)/len(rows),5000)
        self.assertNotIn("question_quality",rows[0]); self.assertNotIn("question_generation",rows[0])
        self.assertEqual(rows[0]["resources"]["images"],[])
        self.assertEqual(rows[1]["resources"]["images"][0]["path"],"/tmp/x.png")
        self.assertTrue(all(audit["format_version"]==QUESTION_AUDIT_FORMAT_VERSION for audit in audits))
        self.assertEqual(audits[0]["question_generation"]["model"],"writer")
        self.assertEqual(audits[0]["question_contract_version"],QUESTION_CONTRACT_VERSION)

    def test_direct_writer_retries_invalid_json_without_template_fallback(self):
        response={"user_request":"我们准备把这批序列接入日常预测。在低算力约束下，请判断字段角色和时间索引是否适合安全建模，并决定需要检查哪些证据来排除泄漏或不规则间隔风险，同时说明何时需要转换格式。","decision_points":["判断数据是否适用","决定检查证据与转换条件"],"constraint_key":"compute_budget","business_facts_used":["decision_constraints.compute_budget"]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); specs=root/"specs.jsonl"; output=root/"questions.jsonl"; rejected=root/"rejected.jsonl"; prompt=root/"writer.txt"
            write_jsonl(specs,[self._question_spec("q1","text_only","q1")]); prompt.write_text("只输出JSON",encoding="utf-8")
            replies=[("not json",{},.1),(json.dumps(response,ensure_ascii=False),{"total_tokens":8},.1)]
            with patch("api_sft.questions.OpenAICompatibleClient.complete",side_effect=replies) as complete:
                rows=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt,max_attempts=2)
            self.assertEqual(complete.call_count,2); self.assertEqual(len(rows),1); self.assertFalse(rejected.exists())

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); specs=root/"specs.jsonl"; output=root/"questions.jsonl"; rejected=root/"rejected.jsonl"; prompt=root/"writer.txt"
            write_jsonl(specs,[self._question_spec("q1","text_only","q1")]); prompt.write_text("只输出JSON",encoding="utf-8")
            with patch("api_sft.questions.OpenAICompatibleClient.complete",return_value=("not json",{},.1)):
                rows=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt,max_attempts=2)
            self.assertEqual(rows,[]); self.assertTrue(rejected.exists()); self.assertNotIn("coverage_template",rejected.read_text(encoding="utf-8"))

    def test_question_resume_invalidates_changed_spec_hash(self):
        first={"user_request":"在低算力约束下，请判断字段角色与时间索引是否适合后续分析，并确定需要检查哪些证据来排除泄漏，同时说明何时需要转换数据格式。","decision_points":["判断数据是否适用","决定补充检查"],"constraint_key":"compute_budget","business_facts_used":["decision_constraints.compute_budget"]}
        second={"user_request":"在可解释性要求下，请重新判断字段角色与时间索引是否适合后续分析，并决定需要检查哪些证据来排除泄漏或不规则间隔风险，同时说明何时有必要转换格式。","decision_points":["重新判断数据是否适用","决定检查证据与转换条件"],"constraint_key":"interpretability","business_facts_used":["decision_constraints.interpretability"]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); specs=root/"specs.jsonl"; output=root/"questions.jsonl"; rejected=root/"rejected.jsonl"; prompt=root/"writer.txt"; prompt.write_text("只输出JSON")
            spec=self._question_spec("q1","text_only","q1"); write_jsonl(specs,[spec])
            with patch("api_sft.questions.OpenAICompatibleClient.complete",return_value=(json.dumps(first,ensure_ascii=False),{},.1)):
                original=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt)
            spec["question_spec_hash"]="changed_hash"; write_jsonl(specs,[spec])
            with patch("api_sft.questions.OpenAICompatibleClient.complete",return_value=(json.dumps(second,ensure_ascii=False),{},.1)) as complete:
                updated=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt,resume=True)
        self.assertEqual(complete.call_count,1); self.assertNotEqual(original[0]["prompt"]["user_request"],updated[0]["prompt"]["user_request"]); self.assertEqual(updated[0]["spec_ref"]["hash"],"changed_hash")

    def test_question_resume_reuses_valid_runtime_and_audit(self):
        response={"user_request":"在低算力约束下，请判断字段角色与时间索引是否适合后续分析，并确定需要检查哪些证据来排除泄漏，同时说明何时需要转换数据格式。","decision_points":["判断数据是否适用","决定补充检查"],"constraint_key":"compute_budget","business_facts_used":["decision_constraints.compute_budget"]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); specs=root/"specs.jsonl"; output=root/"questions.jsonl"; rejected=root/"rejected.jsonl"; prompt=root/"writer.txt"; prompt.write_text("只输出JSON")
            write_jsonl(specs,[self._question_spec("q1","text_only","q1")])
            with patch("api_sft.questions.OpenAICompatibleClient.complete",return_value=(json.dumps(response,ensure_ascii=False),{},.1)):
                first=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt)
            with patch("api_sft.questions.OpenAICompatibleClient.complete") as complete:
                second=generate_questions(specs,output,rejected,{"base_url":"https://example/v1","model":"writer","api_key":"secret"},prompt,resume=True)
            audit=json.loads((root/"questions.audit.jsonl").read_text())
        self.assertEqual(first,second)
        self.assertEqual(complete.call_count,0)
        self.assertEqual(audit["question_contract_version"],QUESTION_CONTRACT_VERSION)

    def test_question_spec_resume_invalidates_changed_scenario_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scenario_root=root/"scenario_files"; archetype=next(item for item in ARCHETYPES if item["name"]=="trend_period_profile"); record=persist_scenario(create_scenario(1,archetype,123),scenario_root)
            scenarios=root/"scenarios.jsonl"; tools=root/"tools.json"; models=root/"models.json"; output=root/"specs.jsonl"
            write_jsonl(scenarios,[record]); write_json(tools,{"tools":[]}); write_json(models,{"models":[]})
            first=build_question_specs(scenarios,tools,models,output,questions_per_scenario=1,task_pool_path=ROOT/"api_sft"/"task_pool.yaml")
            changed=dict(record); changed["scenario_hash"]="new_scenario_hash"; write_jsonl(scenarios,[changed])
            second=build_question_specs(scenarios,tools,models,output,questions_per_scenario=1,resume=True,task_pool_path=ROOT/"api_sft"/"task_pool.yaml")
        self.assertEqual({row["scenario_hash"] for row in second},{"new_scenario_hash"})
        self.assertNotEqual({row["question_spec_hash"] for row in first},{row["question_spec_hash"] for row in second})

    def test_question_specs_reject_v2_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scenarios=root/"scenarios.jsonl"; tools=root/"tools.json"; models=root/"models.json"
            write_jsonl(scenarios,[{"id":"scenario_old","generator_version":"v2"}]); write_json(tools,{"tools":[]}); write_json(models,{"models":[]})
            with self.assertRaisesRegex(RuntimeError,"requires generator v4"):
                build_question_specs(scenarios,tools,models,root/"specs.jsonl")

    def test_unseen_number_and_answer_blueprint_are_rejected(self):
        spec=self._question_spec("q1","text_only","pair")
        metadata={"decision_points":["a","b"],"constraint_key":"compute_budget","business_facts_used":["decision_constraints.compute_budget"]}
        quality=validate_question("请按照以下步骤分析，并给出 Top-3 候选；数据一共有999行，最后完成模型选择和风险判断。"*2,spec,metadata)
        self.assertFalse(quality["passed"]); self.assertFalse(quality["checks"]["no_answer_blueprint"]); self.assertFalse(quality["checks"]["grounded_numbers"])

    def test_observed_data_findings_must_be_questions_not_assertions(self):
        spec=self._question_spec("q1","text_only","pair")
        metadata={"decision_points":["判断缺失","决定处理"],"constraint_key":"compute_budget","business_facts_used":["decision_constraints.compute_budget"]}
        asserted=validate_question("在低算力约束下，这批序列都有不同程度的缺失，请决定插补方式并评估后续分析风险，同时给出处理建议。"*2,spec,metadata)
        residual_asserted=validate_question("在低算力约束下，模型残差均值偏离零，请评估校准方式及其风险，并决定是否需要重新验证。"*2,spec,metadata)
        uncertain=validate_question("在低算力约束下，请判断这批序列是否存在缺失及其位置模式，并决定是否需要插补，同时评估不同处理对后续分析的风险。"*2,spec,metadata)
        self.assertFalse(asserted["checks"]["no_derived_findings"])
        self.assertFalse(residual_asserted["checks"]["no_derived_findings"])
        self.assertTrue(uncertain["checks"]["no_derived_findings"])


class ApiAndVerificationTests(unittest.TestCase):
    def test_api_key_can_come_from_yaml_or_environment(self):
        self.assertEqual(resolve_api_key({"api_key":"yaml-secret"}),"yaml-secret")
        with patch.dict("os.environ",{"MODEL_API_KEY":"env-secret"}):
            self.assertEqual(resolve_api_key({"api_key_env":"MODEL_API_KEY","api_key":"yaml-secret"}),"env-secret")

    def test_dotenv_loading_and_secret_like_env_name_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/".env"; path.write_text("# comment\nDOTENV_TEST_KEY='dotenv-secret'\n",encoding="utf-8")
            with patch.dict("os.environ",{},clear=True):
                self.assertEqual(load_env_file(path),1)
                self.assertEqual(resolve_api_key({"api_key_env":"DOTENV_TEST_KEY"}),"dotenv-secret")
        with self.assertRaisesRegex(RuntimeError,"appears to contain a secret") as caught:
            resolve_api_key({"api_key_env":"sk-do-not-echo-this-value"})
        self.assertNotIn("sk-do-not-echo",str(caught.exception))

    def test_json_fence_and_multimodal_payload(self):
        self.assertEqual(parse_json_object("```json\n{\"a\":1}\n```"),{"a":1})
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/"x.png"; image.write_bytes(b"fake")
            msg=user_message("q",[str(image)])
        self.assertEqual(msg["content"][1]["type"],"image_url")
        self.assertTrue(msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_api_retries_transient_http_error(self):
        calls={"count":0}
        class FakeResponse:
            def raise_for_status(self):
                if calls["count"]==1: raise httpx.HTTPStatusError("429",request=httpx.Request("POST","https://x"),response=httpx.Response(429))
            def json(self): return {"choices":[{"message":{"content":"{\"ok\": true}"}}],"usage":{"total_tokens":3}}
        class FakeClient:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def post(self,*args,**kwargs): calls["count"]+=1; return FakeResponse()
        cfg={"base_url":"https://example/v1","model":"vlm","api_key_env":"TEST_API_KEY","retries":2}
        with patch.dict("os.environ",{"TEST_API_KEY":"secret"}),patch("api_sft.api_client.httpx.Client",FakeClient),patch("api_sft.api_client.time.sleep"):
            raw,usage,_=OpenAICompatibleClient(cfg).complete([{"role":"user","content":"q"}])
        self.assertEqual(calls["count"],2); self.assertEqual(parse_json_object(raw),{"ok":True}); self.assertEqual(usage["total_tokens"],3)

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError): parse_json_object("not json")

    def test_legacy_answer_generation_is_blocked_for_tool_execution_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); questions=root/"questions.jsonl"; output=root/"answers.jsonl"
            write_jsonl(questions,[{"id":"q1","question_spec_version":"3.0","trajectory_requirement":"tool_execution"}])
            with self.assertRaisesRegex(RuntimeError,"require real tool-execution trajectories"):
                generate_answers(questions,output,[])
            self.assertFalse(output.exists())

    def test_deterministic_review_rejects_fake_tool(self):
        answer={"analysis_mode":"text_only","evidence_from_text":["摘要"],"evidence_from_images":[],"tool_plan":[{"step":1,"tool":"fake_tool","reason":"x","branch_condition":"stop"}],"final_answer":"足够长的答案"*80,"model_recommendations":[],"validation_plan":{},"risks_and_uncertainties":["风险"]}
        record={"status":"ok","question_record":{"input_mode":"text_only"},"selection":{"final_answer":answer}}
        review=deterministic_review(record,{"data_profile"},set())
        self.assertFalse(review["passed"]); self.assertTrue(any(x.startswith("invalid_tools") for x in review["flags"]))

    def test_deterministic_review_requires_subtask_relevant_tool(self):
        answer={"analysis_mode":"text_only","evidence_from_text":["摘要"],"evidence_from_images":[],"tool_plan":[{"step":1,"tool":"summary_stats","reason":"x","branch_condition":"stop"}],"final_answer":"足够长的答案"*80,"model_recommendations":[],"validation_plan":{},"risks_and_uncertainties":["风险"]}
        record={"status":"ok","question_record":{"input_mode":"text_only","primary_tools":["dtw_distance"]},"selection":{"final_answer":answer}}
        review=deterministic_review(record,{"summary_stats","dtw_distance"},set())
        self.assertFalse(review["passed"]); self.assertIn("tool_plan_missing_subtask_relevant_tool",review["flags"])

    def test_export_does_not_include_hidden_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); verified=root/"verified.jsonl"; image=root/"i.png"; image.write_bytes(b"png")
            answer={"analysis_mode":"image_text","evidence_from_text":[],"evidence_from_images":["shape"],"tool_plan":[],"final_answer":"answer","model_recommendations":[],"validation_plan":{},"risks_and_uncertainties":[]}
            q={"question":"question","images":[str(image)],"task":"data_profile","task_goal":"diagnosis","series_count":"single","history_length":"long","input_mode":"image_text","recommended_mode":"image_text","image_value":"high","difficulty":"hard"}
            write_jsonl(verified,[{"id":"x","question_record":q,"selection":{"final_answer":answer},"candidates":[],"truth_verifier_review":{"notes":"no secret"}}])
            export_datasets(verified,root/"export"); trl=json.loads((root/"export"/"train_trl_messages.jsonl").read_text())
            self.assertEqual(trl["messages"][1]["content"][0],{"type":"image"}); self.assertNotIn("ground_truth",json.dumps(trl))


if __name__=="__main__": unittest.main()
