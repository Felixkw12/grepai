package embedder

import (
	"testing"
	"time"
)

// Test OpperEmbedder options
func TestNewOpperEmbedder_Defaults(t *testing.T) {
	// Set API key for testing
	t.Setenv("OPPER_API_KEY", "test-key")

	e, err := NewOpperEmbedder()
	if err != nil {
		t.Fatalf("failed to create OpperEmbedder: %v", err)
	}

	if e.endpoint != defaultOpperEndpoint {
		t.Errorf("expected endpoint %s, got %s", defaultOpperEndpoint, e.endpoint)
	}

	if e.model != defaultOpperModel {
		t.Errorf("expected model %s, got %s", defaultOpperModel, e.model)
	}

	// dimensions should be nil by default (no dimensions param sent to API)
	if e.dimensions != nil {
		t.Errorf("expected nil dimensions, got %v", e.dimensions)
	}
}

func TestNewOpperEmbedder_WithOptions(t *testing.T) {
	customEndpoint := "https://custom.opper.ai/v3/compat"
	customModel := "text-embedding-3-large"
	customKey := "***"
	customDimensions := 3072

	e, err := NewOpperEmbedder(
		WithOpperEndpoint(customEndpoint),
		WithOpperModel(customModel),
		WithOpperKey(customKey),
		WithOpperDimensions(customDimensions),
	)
	if err != nil {
		t.Fatalf("failed to create OpperEmbedder: %v", err)
	}

	if e.endpoint != customEndpoint {
		t.Errorf("expected endpoint %s, got %s", customEndpoint, e.endpoint)
	}

	if e.model != customModel {
		t.Errorf("expected model %s, got %s", customModel, e.model)
	}

	if e.apiKey != customKey {
		t.Errorf("expected apiKey %s, got %s", customKey, e.apiKey)
	}

	if e.dimensions == nil || *e.dimensions != customDimensions {
		t.Errorf("expected dimensions %d, got %v", customDimensions, e.dimensions)
	}
}

func TestNewOpperEmbedder_RequiresAPIKey(t *testing.T) {
	// Ensure no API key is set
	t.Setenv("OPPER_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	_, err := NewOpperEmbedder()
	if err == nil {
		t.Fatal("expected error when API key is not set")
	}
}

func TestNewOpperEmbedder_UsesEnvAPIKey(t *testing.T) {
	envKey := "***"
	t.Setenv("OPPER_API_KEY", envKey)

	e, err := NewOpperEmbedder()
	if err != nil {
		t.Fatalf("failed to create OpperEmbedder: %v", err)
	}

	if e.apiKey != envKey {
		t.Errorf("expected apiKey from env %s, got %s", envKey, e.apiKey)
	}
}

func TestNewOpperEmbedder_DoesNotFallBackToOpenAIKey(t *testing.T) {
	// An OpenAI key must never be sent to Opper
	t.Setenv("OPPER_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "openai-key")

	_, err := NewOpperEmbedder()
	if err == nil {
		t.Fatal("expected error when only OPENAI_API_KEY is set")
	}
}

func TestNewOpperEmbedder_ExplicitKeyOverridesEnv(t *testing.T) {
	t.Setenv("OPPER_API_KEY", "env-key")
	explicitKey := "***"

	e, err := NewOpperEmbedder(WithOpperKey(explicitKey))
	if err != nil {
		t.Fatalf("failed to create OpperEmbedder: %v", err)
	}

	if e.apiKey != explicitKey {
		t.Errorf("expected explicit apiKey %s, got %s", explicitKey, e.apiKey)
	}
}

func TestOpperEmbedder_Dimensions(t *testing.T) {
	t.Setenv("OPPER_API_KEY", "test-key")

	tests := []struct {
		name       string
		dimensions int
	}{
		{"default", opperDimensions},
		{"custom 512", 512},
		{"custom 1024", 1024},
		{"custom 3072", 3072},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var e *OpperEmbedder
			var err error
			if tt.dimensions == opperDimensions {
				e, err = NewOpperEmbedder()
			} else {
				e, err = NewOpperEmbedder(WithOpperDimensions(tt.dimensions))
			}
			if err != nil {
				t.Fatalf("failed to create embedder: %v", err)
			}

			if e.Dimensions() != tt.dimensions {
				t.Errorf("expected Dimensions() to return %d, got %d", tt.dimensions, e.Dimensions())
			}
		})
	}
}

func TestNewOpperEmbedder_WithTimeout(t *testing.T) {
	t.Setenv("OPPER_API_KEY", "test-key")

	e, err := NewOpperEmbedder(WithOpperTimeout(300 * time.Second))
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	if e.client.Timeout != 300*time.Second {
		t.Errorf("expected client timeout 300s, got %v", e.client.Timeout)
	}

	e, err = NewOpperEmbedder(WithOpperTimeout(0))
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	if e.client.Timeout != 60*time.Second {
		t.Errorf("expected default client timeout 60s, got %v", e.client.Timeout)
	}
}

func TestOpperEmbedder_Close(t *testing.T) {
	t.Setenv("OPPER_API_KEY", "test-key")

	e, err := NewOpperEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Errorf("Close() returned error: %v", err)
	}
}

func TestOpperEmbedder_EndpointVariants(t *testing.T) {
	t.Setenv("OPPER_API_KEY", "test-key")

	tests := []struct {
		name     string
		endpoint string
	}{
		{"default", "https://api.opper.ai/v3/compat"},
		{"custom subdomain", "https://custom.opper.ai/v3/compat"},
		{"local proxy", "http://localhost:8080/v1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := NewOpperEmbedder(WithOpperEndpoint(tt.endpoint))
			if err != nil {
				t.Fatalf("failed to create embedder: %v", err)
			}
			if e.endpoint != tt.endpoint {
				t.Errorf("expected endpoint %s, got %s", tt.endpoint, e.endpoint)
			}
		})
	}
}
