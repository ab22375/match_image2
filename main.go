package main

import (
	"database/sql"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"gocv.io/x/gocv"
)

// ImageInfo holds the image metadata and features
type ImageInfo struct {
	ID             int64  `json:"id"`
	Path           string `json:"path"`
	Width          int    `json:"width"`
	Height         int    `json:"height"`
	CreatedAt      string `json:"created_at"`
	ModifiedAt     string `json:"modified_at"`
	Size           int64  `json:"size"`
	AverageHash    string `json:"average_hash"`
	PerceptualHash string `json:"perceptual_hash"`
}

// ImageMatch holds the similarity scores
type ImageMatch struct {
	Path      string
	SSIMScore float64
}

// Database operations
func initDatabase(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}

	// Create table if it doesn't exist
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS images (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		path TEXT NOT NULL UNIQUE,
		width INTEGER,
		height INTEGER,
		created_at TEXT,
		modified_at TEXT,
		size INTEGER,
		average_hash TEXT,
		perceptual_hash TEXT,
		features BLOB
	);
	CREATE INDEX IF NOT EXISTS idx_path ON images(path);
	CREATE INDEX IF NOT EXISTS idx_average_hash ON images(average_hash);
	CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON images(perceptual_hash);`

	_, err = db.Exec(createTableSQL)
	return db, err
}

// Load image in grayscale with error handling
func loadImage(path string) (gocv.Mat, error) {
	img := gocv.IMRead(path, gocv.IMReadGrayScale)
	if img.Empty() {
		return img, fmt.Errorf("failed to load image: %s", path)
	}
	return img, nil
}

// Compute average hash for image indexing
func computeAverageHash(img gocv.Mat) (string, error) {
	// Resize to 8x8
	resized := gocv.NewMat()
	defer resized.Close()

	gocv.Resize(img, &resized, image.Point{X: 8, Y: 8}, 0, 0, gocv.InterpolationArea)

	// Convert to grayscale if not already
	gray := gocv.NewMat()
	defer gray.Close()

	if img.Channels() > 1 {
		gocv.CvtColor(resized, &gray, gocv.ColorBGRToGray)
	} else {
		resized.CopyTo(&gray)
	}

	// Calculate average pixel value manually
	var sum float64
	totalPixels := gray.Rows() * gray.Cols()

	for i := 0; i < gray.Rows(); i++ {
		for j := 0; j < gray.Cols(); j++ {
			sum += float64(gray.GetUCharAt(i, j))
		}
	}

	threshold := sum / float64(totalPixels)

	// Compute the hash
	var hash strings.Builder
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			pixel := gray.GetUCharAt(i, j)
			if float64(pixel) >= threshold {
				hash.WriteString("1")
			} else {
				hash.WriteString("0")
			}
		}
	}

	return hash.String(), nil
}

// Compute perceptual hash (pHash) for better matching
func computePerceptualHash(img gocv.Mat) (string, error) {
	// Resize to 32x32
	resized := gocv.NewMat()
	defer resized.Close()

	gocv.Resize(img, &resized, image.Point{X: 32, Y: 32}, 0, 0, gocv.InterpolationArea)

	// Convert to grayscale if not already
	gray := gocv.NewMat()
	defer gray.Close()

	if img.Channels() > 1 {
		gocv.CvtColor(resized, &gray, gocv.ColorBGRToGray)
	} else {
		resized.CopyTo(&gray)
	}

	// Since DCT isn't available, let's implement a simplified alternative hash
	// We'll use a variation of the average hash with more regions

	// Create a simplified hash based on brightness patterns
	regions := 8 // 8x8 regions
	var regionValues []float32

	// Calculate average brightness in each region
	regionHeight := gray.Rows() / regions
	regionWidth := gray.Cols() / regions

	for i := 0; i < regions; i++ {
		for j := 0; j < regions; j++ {
			// Calculate region boundaries
			startY := i * regionHeight
			endY := (i + 1) * regionHeight
			startX := j * regionWidth
			endX := (j + 1) * regionWidth

			// Calculate average for region
			var sum float32
			var count int
			for y := startY; y < endY; y++ {
				for x := startX; x < endX; x++ {
					sum += float32(gray.GetUCharAt(y, x))
					count++
				}
			}

			avg := sum / float32(count)
			regionValues = append(regionValues, avg)
		}
	}

	// We've already computed the region values in the code above
	// No need to extract additional values

	// Calculate median
	sort.Slice(regionValues, func(i, j int) bool {
		return regionValues[i] < regionValues[j]
	})
	median := regionValues[len(regionValues)/2]

	// Create hash based on whether each value is above median
	var hash strings.Builder
	for _, val := range regionValues {
		if val > median {
			hash.WriteString("1")
		} else {
			hash.WriteString("0")
		}
	}

	return hash.String(), nil
}

// Simplified and more robust SSIM implementation
func computeSSIM(img1, img2 gocv.Mat) float64 {
	// Check for valid matrices
	if img1.Empty() || img2.Empty() || img1.Rows() == 0 || img1.Cols() == 0 ||
		img2.Rows() == 0 || img2.Cols() == 0 {
		return 0.0
	}

	// Convert to 8-bit grayscale if needed
	img1Gray := gocv.NewMat()
	img2Gray := gocv.NewMat()
	defer img1Gray.Close()
	defer img2Gray.Close()

	if img1.Type() != gocv.MatTypeCV8U {
		img1.ConvertTo(&img1Gray, gocv.MatTypeCV8U)
	} else {
		img1.CopyTo(&img1Gray)
	}

	if img2.Type() != gocv.MatTypeCV8U {
		img2.ConvertTo(&img2Gray, gocv.MatTypeCV8U)
	} else {
		img2.CopyTo(&img2Gray)
	}

	// Ensure images are same size
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(img2Gray, &resized, image.Point{X: img1Gray.Cols(), Y: img1Gray.Rows()}, 0, 0, gocv.InterpolationLinear)

	// Calculate simple mean difference
	diff := gocv.NewMat()
	defer diff.Close()

	gocv.AbsDiff(img1Gray, resized, &diff)

	mean := gocv.NewMat()
	stdDev := gocv.NewMat()
	defer mean.Close()
	defer stdDev.Close()

	if diff.Empty() || diff.Rows() == 0 || diff.Cols() == 0 {
		return 0.0
	}

	gocv.MeanStdDev(diff, &mean, &stdDev)

	if mean.Empty() || mean.Rows() == 0 || mean.Cols() == 0 {
		return 0.0
	}

	// Calculate similarity score (1 - normalized difference)
	meanDiff := mean.GetDoubleAt(0, 0)
	if meanDiff > 255.0 {
		return 0.0
	}

	// Return similarity score (1 = identical, 0 = completely different)
	return 1.0 - (meanDiff / 255.0)
}

// Process and store image information in the database
func processAndStoreImage(db *sql.DB, path string, wg *sync.WaitGroup, errChan chan<- error, semaphore chan struct{}) {
	defer wg.Done()
	defer func() { <-semaphore }() // Release semaphore when done

	// Check if image already exists in database
	var count int
	err := db.QueryRow("SELECT COUNT(*) FROM images WHERE path = ?", path).Scan(&count)
	if err != nil {
		errChan <- fmt.Errorf("database error for %s: %v", path, err)
		return
	}

	if count > 0 {
		// Image already indexed, check if it needs update
		fileInfo, err := os.Stat(path)
		if err != nil {
			errChan <- fmt.Errorf("cannot stat file %s: %v", path, err)
			return
		}

		var storedModTime string
		err = db.QueryRow("SELECT modified_at FROM images WHERE path = ?", path).Scan(&storedModTime)
		if err != nil {
			errChan <- fmt.Errorf("cannot get modified time for %s: %v", path, err)
			return
		}

		// Parse stored time and compare with file modified time
		storedTime, err := time.Parse(time.RFC3339, storedModTime)
		if err != nil {
			errChan <- fmt.Errorf("cannot parse stored time for %s: %v", path, err)
			return
		}

		// If file hasn't been modified, skip processing
		if !fileInfo.ModTime().After(storedTime) {
			return
		}
	}

	// Load and process the image
	img, err := loadImage(path)
	if err != nil {
		errChan <- err
		return
	}
	defer img.Close()

	// Get file info
	fileInfo, err := os.Stat(path)
	if err != nil {
		errChan <- fmt.Errorf("cannot stat file %s: %v", path, err)
		return
	}

	// Compute hashes
	avgHash, err := computeAverageHash(img)
	if err != nil {
		errChan <- fmt.Errorf("cannot compute average hash for %s: %v", path, err)
		return
	}

	pHash, err := computePerceptualHash(img)
	if err != nil {
		errChan <- fmt.Errorf("cannot compute perceptual hash for %s: %v", path, err)
		return
	}

	// Store in database
	now := time.Now().Format(time.RFC3339)
	modTime := fileInfo.ModTime().Format(time.RFC3339)

	// Prepare statement to avoid SQL injection
	stmt, err := db.Prepare(`
		INSERT OR REPLACE INTO images (
			path, width, height, created_at, modified_at, size, average_hash, perceptual_hash
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`)
	if err != nil {
		errChan <- fmt.Errorf("cannot prepare statement for %s: %v", path, err)
		return
	}
	defer stmt.Close()

	_, err = stmt.Exec(
		path,
		img.Cols(),
		img.Rows(),
		now,
		modTime,
		fileInfo.Size(),
		avgHash,
		pHash,
	)

	if err != nil {
		errChan <- fmt.Errorf("cannot insert data for %s: %v", path, err)
		return
	}
}

// Scan folder and store image information in database
func scanAndStoreFolder(db *sql.DB, folderPath string) error {
	var wg sync.WaitGroup

	// Channel for collecting errors
	errorsChan := make(chan error, 100)

	// Semaphore to limit concurrent goroutines
	semaphore := make(chan struct{}, 8)

	// Count total files before starting
	var totalFiles int
	filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
			totalFiles++
		}
		return nil
	})

	fmt.Printf("Starting image indexing...\nTotal image files to process: %d\n", totalFiles)

	// Create a ticker for progress indicator
	ticker := time.NewTicker(500 * time.Millisecond)
	done := make(chan bool)
	processed := 0
	var mu sync.Mutex

	go func() {
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				mu.Lock()
				fmt.Printf("\rProgress: %d/%d", processed, totalFiles)
				mu.Unlock()
			}
		}
	}()

	// Process files
	err := filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}

		// Skip non-image files
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			return nil
		}

		wg.Add(1)
		// Acquire semaphore
		semaphore <- struct{}{}

		go func(p string) {
			processAndStoreImage(db, p, &wg, errorsChan, semaphore)
			mu.Lock()
			processed++
			mu.Unlock()
		}(path)

		return nil
	})

	// Wait for all goroutines to complete
	wg.Wait()
	close(errorsChan)

	// Stop the progress indicator
	ticker.Stop()
	done <- true
	fmt.Println("\nIndexing complete.")

	// Check for errors
	var errorCount int
	for err := range errorsChan {
		log.Printf("Warning: %v", err)
		errorCount++
	}

	if errorCount > 0 {
		fmt.Printf("Encountered %d errors during indexing.\n", errorCount)
	}

	return err
}

// Find similar images in the database
func findSimilarImages(db *sql.DB, queryPath string, threshold float64) ([]ImageMatch, error) {
	queryImg, err := loadImage(queryPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load query image: %v", err)
	}
	defer queryImg.Close()

	// Compute hashes for query image
	avgHash, err := computeAverageHash(queryImg)
	if err != nil {
		return nil, fmt.Errorf("cannot compute average hash: %v", err)
	}

	pHash, err := computePerceptualHash(queryImg)
	if err != nil {
		return nil, fmt.Errorf("cannot compute perceptual hash: %v", err)
	}

	// Use hamming distance to find potential matches
	// Converting binary hash to integer and using bit operations would be more efficient
	// but for simplicity we'll use string comparison here

	// Query database for potential matches based on hash similarity
	rows, err := db.Query(`
		SELECT path, average_hash, perceptual_hash FROM images
	`)
	if err != nil {
		return nil, fmt.Errorf("database query error: %v", err)
	}
	defer rows.Close()

	var matches []ImageMatch
	var wg sync.WaitGroup
	var mutex sync.Mutex
	semaphore := make(chan struct{}, 8)

	for rows.Next() {
		var path, dbAvgHash, dbPHash string
		err := rows.Scan(&path, &dbAvgHash, &dbPHash)
		if err != nil {
			return nil, fmt.Errorf("error scanning row: %v", err)
		}

		// Calculate hamming distance (number of different bits)
		var avgHashDistance, pHashDistance int
		for i := 0; i < len(avgHash) && i < len(dbAvgHash); i++ {
			if avgHash[i] != dbAvgHash[i] {
				avgHashDistance++
			}
		}

		for i := 0; i < len(pHash) && i < len(dbPHash); i++ {
			if pHash[i] != dbPHash[i] {
				pHashDistance++
			}
		}

		// If hash distance is within threshold, compute SSIM for more accurate comparison
		if avgHashDistance <= 10 || pHashDistance <= 12 { // Adjustable thresholds
			wg.Add(1)
			semaphore <- struct{}{}

			go func(p string) {
				defer wg.Done()
				defer func() { <-semaphore }()

				// Load candidate image and compute SSIM
				candidateImg, err := loadImage(p)
				if err != nil {
					log.Printf("Warning: Failed to load candidate image %s: %v", p, err)
					return
				}
				defer candidateImg.Close()

				ssimScore := computeSSIM(queryImg, candidateImg)

				// If SSIM score is above threshold, add to matches
				if ssimScore >= threshold {
					match := ImageMatch{
						Path:      p,
						SSIMScore: ssimScore,
					}

					mutex.Lock()
					matches = append(matches, match)
					mutex.Unlock()
				}
			}(path)
		}
	}

	wg.Wait()

	// Sort matches by SSIM score (higher is better)
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].SSIMScore > matches[j].SSIMScore
	})

	return matches, nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("Usage:\n")
		fmt.Printf("  %s scan <folder_path> [db_path]\n", os.Args[0])
		fmt.Printf("  %s search <query_image_path> [db_path] [threshold]\n", os.Args[0])
		os.Exit(1)
	}

	command := os.Args[1]
	dbPath := "images.db"

	switch command {
	case "scan":
		if len(os.Args) < 3 {
			fmt.Println("Error: Missing folder path")
			os.Exit(1)
		}

		folderPath := os.Args[2]

		// Set custom database path if provided
		if len(os.Args) > 3 {
			dbPath = os.Args[3]
		}

		// Verify paths exist
		if _, err := os.Stat(folderPath); os.IsNotExist(err) {
			log.Fatalf("Folder path does not exist: %s", folderPath)
		}

		startTime := time.Now()

		// Initialize database
		db, err := initDatabase(dbPath)
		if err != nil {
			log.Fatalf("Error initializing database: %v", err)
		}
		defer db.Close()

		// Scan folder and store image information
		err = scanAndStoreFolder(db, folderPath)
		if err != nil {
			log.Fatalf("Error scanning folder: %v", err)
		}

		// Print execution time
		duration := time.Since(startTime)
		fmt.Printf("\nTotal execution time: %v\n", duration)

	case "search":
		if len(os.Args) < 3 {
			fmt.Println("Error: Missing query image path")
			os.Exit(1)
		}

		queryPath := os.Args[2]

		// Set custom database path if provided
		if len(os.Args) > 3 {
			dbPath = os.Args[3]
		}

		// Set custom threshold if provided
		threshold := 0.8 // Default threshold
		if len(os.Args) > 4 {
			fmt.Sscanf(os.Args[4], "%f", &threshold)
		}

		// Verify paths exist
		if _, err := os.Stat(queryPath); os.IsNotExist(err) {
			log.Fatalf("Query image does not exist: %s", queryPath)
		}

		if _, err := os.Stat(dbPath); os.IsNotExist(err) {
			log.Fatalf("Database does not exist: %s. Run scan command first.", dbPath)
		}

		startTime := time.Now()

		// Open database
		db, err := sql.Open("sqlite3", dbPath)
		if err != nil {
			log.Fatalf("Error opening database: %v", err)
		}
		defer db.Close()

		fmt.Println("Searching for similar images...")

		// Find similar images
		matches, err := findSimilarImages(db, queryPath, threshold)
		if err != nil {
			log.Fatalf("Error finding similar images: %v", err)
		}

		// Print top matches
		fmt.Println("\nTop Matches:")
		limit := 5 // Show top 5 matches
		for i := 0; i < limit && i < len(matches); i++ {
			fmt.Printf("%d. Image: %s\n   SSIM Score: %.4f\n",
				i+1, matches[i].Path, matches[i].SSIMScore)
		}

		// Print execution time
		duration := time.Since(startTime)
		fmt.Printf("\nTotal search time: %v\n", duration)

	default:
		fmt.Printf("Unknown command: %s\n", command)
		fmt.Printf("Usage:\n")
		fmt.Printf("  %s scan <folder_path> [db_path]\n", os.Args[0])
		fmt.Printf("  %s search <query_image_path> [db_path] [threshold]\n", os.Args[0])
		os.Exit(1)
	}
}
