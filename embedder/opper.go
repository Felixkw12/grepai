package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const (
	defaultOpperEndpoint = "https://api.opper.ai/v3/compat"
	defaultOpperModel    = "text-embedding-3-small"
	opperDimensions      = 1536
)

// OpperEmbedder implements the Embedder interface for the Opper API, an
// OpenAI-compatible gateway. Model ids are bare pool names such as
// text-embedding-3-small.
// Note: it sends one batch per request and has no worker pool. If parallel
// processing is needed in the future, consider implementing the BatchEmbedder
// interface similar to OpenAIEmbedder.
type OpperEmbedder struct {
	endpoint   string
	model      string
	apiKey     string
	dimensions *int
	client     *http.Client
}

type opperEmbedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions *int     `json:"dimensions,omitempty"`
}

type opperEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type opperErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

type OpperOption func(*OpperEmbedder)

func WithOpperEndpoint(endpoint string) OpperOption {
	return func(e *OpperEmbedder) {
		e.endpoint = endpoint
	}
}

func WithOpperModel(model string) OpperOption {
	return func(e *OpperEmbedder) {
		e.model = model
	}
}

func WithOpperKey(key string) OpperOption {
	return func(e *OpperEmbedder) {
		e.apiKey = key
	}
}

func WithOpperDimensions(dimensions int) OpperOption {
	return func(e *OpperEmbedder) {
		e.dimensions = &dimensions
	}
}

// WithOpperTimeout overrides the HTTP client timeout for embedding
// requests. Values <= 0 are ignored, preserving the default.
func WithOpperTimeout(d time.Duration) OpperOption {
	return func(e *OpperEmbedder) {
		if d > 0 && e.client != nil {
			e.client.Timeout = d
		}
	}
}

func NewOpperEmbedder(opts ...OpperOption) (*OpperEmbedder, error) {
	e := &OpperEmbedder{
		endpoint:   defaultOpperEndpoint,
		model:      defaultOpperModel,
		dimensions: nil, // nil = let the model use its native dimensions
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(e)
	}

	// Try to get API key from environment if not set. There is deliberately
	// no OPENAI_API_KEY fallback, so an OpenAI key is never sent to Opper.
	if e.apiKey == "" {
		e.apiKey = os.Getenv("OPPER_API_KEY")
	}

	if e.apiKey == "" {
		return nil, fmt.Errorf("opper API key not set (use OPPER_API_KEY environment variable)")
	}

	return e, nil
}

func (e *OpperEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return embeddings[0], nil
}

func (e *OpperEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := opperEmbedRequest{
		Model:      e.model,
		Input:      texts,
		Dimensions: e.dimensions,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/embeddings", e.endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", e.apiKey))

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Opper: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp opperErrorResponse
		msg := string(body)
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			msg = errResp.Error.Message
		}
		return nil, fmt.Errorf("opper API error (status %d): %s", resp.StatusCode, msg)
	}

	var result opperEmbedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(result.Data))
	}

	// Sort by index to maintain order
	embeddings := make([][]float32, len(texts))
	for _, item := range result.Data {
		embeddings[item.Index] = item.Embedding
	}

	return embeddings, nil
}

func (e *OpperEmbedder) Dimensions() int {
	if e.dimensions == nil {
		return opperDimensions
	}
	return *e.dimensions
}

func (e *OpperEmbedder) Close() error {
	return nil
}

// Ping checks if the Opper API is reachable
func (e *OpperEmbedder) Ping(ctx context.Context) error {
	url := fmt.Sprintf("%s/embeddings", e.endpoint)
	pingReq := map[string]string{
		"model": e.model,
		"input": "test",
	}
	jsonData, err := json.Marshal(pingReq)
	if err != nil {
		return fmt.Errorf("failed to marshal ping request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", e.apiKey))

	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to reach Opper at %s: %w", e.endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("opper returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}
