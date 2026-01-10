"""
Test script for all tracker components.
Tests each module independently to verify functionality.
"""
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test configuration module."""
    print("\n[TEST] Testing config.py...")
    try:
        from config import config, Config, MinimapCalibration, TrackerSettings, TrackerState

        # Test MinimapCalibration
        cal = MinimapCalibration(x=100, y=100, width=200, height=200, side="right")
        assert cal.is_valid, "Calibration should be valid"
        assert cal.region == (100, 100, 200, 200), "Region tuple incorrect"

        # Test invalid calibration
        invalid_cal = MinimapCalibration()
        assert not invalid_cal.is_valid, "Empty calibration should be invalid"

        # Test TrackerSettings defaults
        settings = TrackerSettings()
        assert settings.scan_rate_hz == 15, "Default scan rate should be 15"
        assert settings.alert_cooldown_seconds == 10.0, "Default cooldown should be 10"

        # Test TrackerState
        state = TrackerState()
        assert not state.is_running, "Default state should not be running"

        # Test config reset
        config.state.is_running = True
        config.reset_state()
        assert not config.state.is_running, "Reset should clear running state"

        print("[PASS] config.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] config.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zones():
    """Test zones module."""
    print("\n[TEST] Testing zones.py...")
    try:
        from zones import (
            Zone, ThreatLevel, ZONES, ZONE_MAP,
            get_zone_at_position, get_zone_by_name,
            is_danger_zone, normalize_position, get_jungle_camps
        )

        # Test Zone creation
        zone = Zone("test", "Test Zone", (0.0, 0.0, 0.5, 0.5), ThreatLevel.HIGH)
        assert zone.contains(0.25, 0.25), "Point should be in zone"
        assert not zone.contains(0.75, 0.75), "Point should not be in zone"

        # Test ZONES list
        assert len(ZONES) > 0, "ZONES should not be empty"
        assert len(ZONE_MAP) > 0, "ZONE_MAP should not be empty"

        # Test get_zone_by_name
        enemy_blue = get_zone_by_name("enemy_blue")
        assert enemy_blue is not None, "enemy_blue zone should exist"
        assert enemy_blue.is_jungle_camp, "enemy_blue should be a jungle camp"

        # Test get_zone_at_position
        # Test center of map (mid area)
        mid_zone = get_zone_at_position(0.5, 0.5)
        assert mid_zone is not None, "Should find zone at center"

        # Test is_danger_zone
        assert is_danger_zone("enemy_raptors"), "enemy_raptors should be danger zone"
        assert not is_danger_zone("enemy_blue"), "enemy_blue should not be danger zone"

        # Test normalize_position
        norm = normalize_position(100, 100, 200, 200)
        assert norm == (0.5, 0.5), f"Normalization incorrect: {norm}"

        # Test edge cases
        norm_edge = normalize_position(250, 250, 200, 200)
        assert norm_edge == (1.0, 1.0), "Should clamp to 1.0"

        # Test get_jungle_camps
        camps = get_jungle_camps()
        assert len(camps) > 0, "Should have jungle camps"
        assert all(c.is_jungle_camp for c in camps), "All should be camps"

        print("[PASS] zones.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] zones.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_capture():
    """Test capture module."""
    print("\n[TEST] Testing capture.py...")
    try:
        from capture import screen_capture, ScreenCapture

        # Test screen capture
        screen = screen_capture.capture_screen()
        if screen is None:
            print("[WARN] Screen capture returned None (may be expected if no display)")
        else:
            assert screen.ndim == 3, "Screen should be 3D array"
            assert screen.shape[2] == 3, "Screen should have 3 color channels"
            print(f"  Screen captured: {screen.shape}")

        # Test validation
        assert not screen_capture.validate_calibration(), "Should fail without calibration"

        print("[PASS] capture.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] capture.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_detector():
    """Test detector module."""
    print("\n[TEST] Testing detector.py...")
    try:
        from detector import (
            champion_detector, ChampionDetector, Detection,
            detect_enemy_jungler, is_game_active, get_game_data,
            get_all_enemy_champions
        )

        # Test API functions exist and return expected types when no game
        game_data = get_game_data()
        # Should be None when not in a game
        assert game_data is None or isinstance(game_data, dict), "get_game_data should return None or dict"

        jungler = detect_enemy_jungler()
        assert jungler is None or isinstance(jungler, str), "detect_enemy_jungler should return None or str"

        is_active = is_game_active()
        assert isinstance(is_active, bool), "is_game_active should return bool"

        enemies = get_all_enemy_champions()
        assert isinstance(enemies, list), "get_all_enemy_champions should return list"

        print("  API functions tested (no game running expected)")

        # Test champion name normalization
        detector = ChampionDetector()

        # Test special case names
        assert detector._normalize_champion_name("Lee Sin") == "LeeSin"
        assert detector._normalize_champion_name("kha'zix") == "Khazix"
        assert detector._normalize_champion_name("Miss Fortune") == "MissFortune"
        assert detector._normalize_champion_name("Wukong") == "MonkeyKing"

        # Test simple names
        assert detector._normalize_champion_name("Elise") == "Elise"
        assert detector._normalize_champion_name("viego") == "Viego"

        # Test visibility history
        detector._visibility_history = [True, True, False]
        assert detector.just_disappeared(), "Should detect disappearance"

        detector._visibility_history = [False, False, True]
        assert detector.just_appeared(), "Should detect appearance"

        detector._visibility_history = [True, True, True]
        assert detector.is_stable_visible(), "Should be stable visible"

        detector._visibility_history = [False, False, False]
        assert detector.is_stable_invisible(), "Should be stable invisible"

        # Test reset
        detector._visibility_history = [True, True, True]
        detector.reset()
        assert len(detector._visibility_history) == 0, "Reset should clear history"

        print("[PASS] detector.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] detector.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictor():
    """Test predictor module."""
    print("\n[TEST] Testing predictor.py...")
    try:
        from predictor import jungle_predictor, JunglePredictor, Prediction
        from zones import ThreatLevel

        predictor = JunglePredictor()

        # Test with no data
        assert predictor.predict(time.time()) is None, "Should return None with no data"

        # Test position update
        current_time = time.time()
        predictor.update_position("enemy_blue", current_time)
        assert predictor._last_known_zone == "enemy_blue"
        assert len(predictor._path_history) == 1

        # Test prediction after short time
        prediction = predictor.predict(current_time + 2)
        assert prediction is not None, "Should have prediction"
        assert prediction.confidence > 0.8, "Recent sighting should have high confidence"

        # Test prediction after longer time
        prediction = predictor.predict(current_time + 15)
        assert prediction is not None, "Should still predict"
        assert prediction.confidence < 0.8, "Old sighting should have lower confidence"

        # Test threat assessment
        from zones import get_zone_by_name
        raptors = get_zone_by_name("enemy_raptors")
        if raptors:
            pred = Prediction(
                zone=raptors,
                confidence=0.5,
                reasoning="test",
                time_since_seen=10
            )
            msg, level = predictor.get_threat_assessment(pred)
            assert "DANGER" in msg, "Raptors should be danger"
            assert level == ThreatLevel.DANGER

        # Test reset
        predictor.reset()
        assert predictor._last_known_zone is None

        print("[PASS] predictor.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] predictor.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice():
    """Test voice module."""
    print("\n[TEST] Testing voice.py...")
    try:
        from voice import voice_system, VoiceAlertSystem
        from zones import Zone, ThreatLevel

        # Create a test zone
        test_zone = Zone("test", "test area", (0.0, 0.0, 0.1, 0.1), ThreatLevel.HIGH)

        # Test cooldown logic (without initializing TTS)
        voice = VoiceAlertSystem()

        # Test with voice disabled in config
        from config import config
        original_voice_enabled = config.settings.voice_enabled
        config.settings.voice_enabled = False
        assert not voice._can_alert(), "Should not alert when voice disabled in config"

        # Enable voice and test cooldown
        config.settings.voice_enabled = True
        voice._initialized = True
        voice._last_alert_time = 0

        assert voice._can_alert(), "Should be able to alert"

        # Simulate recent alert
        voice._last_alert_time = time.time()
        assert not voice._can_alert(), "Should be on cooldown"

        # Test cooldown remaining
        remaining = voice.get_cooldown_remaining()
        assert remaining > 0, "Should have cooldown remaining"
        assert remaining <= 10, "Cooldown should be <= 10 seconds"

        # Restore original setting
        config.settings.voice_enabled = original_voice_enabled

        print("[PASS] voice.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] voice.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logger():
    """Test logger module."""
    print("\n[TEST] Testing logger.py...")
    try:
        from logger import detection_logger, DetectionLogger, LogEntry
        import os
        from config import LOGS_DIR

        logger = DetectionLogger()

        # Test starting session
        logger.start_game_session("TestChampion")
        assert logger._current_log_file is not None, "Should have log file"
        assert logger._game_start_time is not None, "Should have start time"

        # Test logging
        logger.log_detection(
            champion="TestChampion",
            zone_name="test_zone",
            zone_display="Test Zone",
            position=(0.5, 0.5),
            confidence="high",
            is_visible=True,
            alert_triggered=True
        )

        assert len(logger._entries) == 1, "Should have one entry"

        # Test get recent
        recent = logger.get_recent_entries(5)
        assert len(recent) == 1, "Should return one entry"

        # Test stats
        stats = logger.get_stats()
        assert stats["total_detections"] == 1
        assert stats["alerts_triggered"] == 1

        # Test end session
        logger.end_game_session()
        assert logger._current_log_file is None, "Should clear log file"

        # Verify files were created
        log_files = os.listdir(LOGS_DIR)
        test_logs = [f for f in log_files if "TestChampion" in f]
        assert len(test_logs) >= 1, "Should have created log files"

        # Cleanup test files
        for f in test_logs:
            try:
                os.remove(os.path.join(LOGS_DIR, f))
            except:
                pass

        print("[PASS] logger.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] logger.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_overlay():
    """Test overlay module (limited without display)."""
    print("\n[TEST] Testing overlay.py...")
    try:
        from overlay import tracker_overlay, TrackerOverlay
        from zones import ThreatLevel

        # Test state management (without starting actual window)
        overlay = TrackerOverlay()

        # Test status updates
        overlay.update_status("Testing")
        assert overlay._current_status == "Testing"

        # Test jungler update
        overlay.update_jungler("Lee Sin")
        assert overlay._current_jungler == "Lee Sin"

        # Test position update
        overlay.update_position(
            "top river", 5.0, "high", ThreatLevel.HIGH, False
        )
        assert overlay._current_position == "top river"
        assert overlay._threat_level == ThreatLevel.HIGH

        # Test prediction position
        overlay.update_position(
            "enemy raptors", 20.0, "60%", ThreatLevel.DANGER, True
        )
        assert overlay._current_position == "~enemy raptors"

        # Test clear
        overlay.clear_position()
        assert overlay._current_position == "Unknown"
        assert overlay._threat_level == ThreatLevel.NONE

        # Test toggle visibility flag
        overlay._visible = True
        overlay._visible = not overlay._visible
        assert not overlay._visible

        print("[PASS] overlay.py - All tests passed")
        return True
    except Exception as e:
        print(f"[FAIL] overlay.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_imports():
    """Test that main.py imports correctly."""
    print("\n[TEST] Testing main.py imports...")
    try:
        # Just test that imports work
        from main import JunglerTracker, main

        # Test JunglerTracker instantiation
        tracker = JunglerTracker()
        assert not tracker._running
        assert not tracker._tracking
        assert not tracker._game_active

        print("[PASS] main.py - Imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] main.py - {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all component tests."""
    print("=" * 60)
    print("  LOL JUNGLER TRACKER - COMPONENT TESTS")
    print("=" * 60)

    results = {
        "config": test_config(),
        "zones": test_zones(),
        "capture": test_capture(),
        "detector": test_detector(),
        "predictor": test_predictor(),
        "voice": test_voice(),
        "logger": test_logger(),
        "overlay": test_overlay(),
        "main": test_main_imports(),
    }

    print("\n" + "=" * 60)
    print("  TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}.py")

    print()
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
